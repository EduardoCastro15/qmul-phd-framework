import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
import csv
from torch.utils.data import DataLoader
from sklearn import metrics
sys.path.append('%s/pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *

# python src/python/run_foodweb_seal.py \
#   --foodweb-csv src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv \
#   --mat-folder src/matlab/data/foodwebs_mat \
#   --train-ratio 0.9 \
#   --num-experiments 2 \
#   --hop 1 \
#   --base-seed 12345 \
#   --continue-on-error


def safe_divide(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def compute_link_prediction_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(np.unique(labels)) < 2:
        roc_auc = np.nan
        pr_auc = np.nan
    else:
        roc_auc = metrics.roc_auc_score(labels, scores)
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(
            labels, scores, pos_label=1
        )
        pr_auc = metrics.auc(recall_curve[::-1], precision_curve[::-1])

    predictions = scores > threshold
    positives = labels == 1
    negatives = labels == 0

    tp = int(np.sum(predictions & positives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    tn = int(np.sum(~predictions & negatives))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1_score = safe_divide(2 * precision * recall, precision + recall)

    return {
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'F1Score': f1_score,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
    }


def remove_self_loops(A):
    A = ssp.csr_matrix(A)
    A = A.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def binarize_adjacency(A):
    A = remove_self_loops(A)
    A.data = np.ones_like(A.data)
    return A


def compute_foodweb_metrics(A):
    A = binarize_adjacency(A)
    n = A.shape[0]
    links = int(A.nnz)
    connectance = safe_divide(links, n * max(0, n - 1))

    in_degree = np.asarray(A.sum(axis=0)).ravel()
    out_degree = np.asarray(A.sum(axis=1)).ravel()
    total_degree = in_degree + out_degree

    basal_mask = (in_degree == 0) & (out_degree > 0)
    top_mask = (in_degree > 0) & (out_degree == 0)
    intermediate_mask = (in_degree > 0) & (out_degree > 0)
    isolate_mask = (in_degree == 0) & (out_degree == 0)

    generality = in_degree
    vulnerability = out_degree
    positive_generality = generality[generality > 0]
    positive_vulnerability = vulnerability[vulnerability > 0]

    mean_generality = float(np.mean(positive_generality)) if positive_generality.size else 0.0
    mean_vulnerability = float(np.mean(positive_vulnerability)) if positive_vulnerability.size else 0.0
    mean_degree = float(np.mean(total_degree)) if n else 0.0

    prey_count_per_consumer = in_degree
    basal_or_isolate_mask = prey_count_per_consumer == 0
    denom = np.maximum(prey_count_per_consumer, 1)
    P = ssp.diags(1.0 / denom).dot(A.transpose()).toarray()
    M = np.eye(n) - P
    trophic_level = np.full(n, np.nan)
    solve_ok = False

    try:
        cond = np.linalg.cond(M)
        rcond = 0.0 if not np.isfinite(cond) or cond == 0 else 1.0 / cond
        if np.isfinite(rcond) and rcond > 1e-10:
            tl_try = np.linalg.solve(M, np.ones(n))
            tl_try[basal_or_isolate_mask] = 1
            if (
                np.all(np.isfinite(tl_try))
                and np.all(tl_try >= 1 - 1e-8)
                and np.all(tl_try <= max(20, n))
            ):
                trophic_level = tl_try
                solve_ok = True
    except Exception:
        solve_ok = False

    if not solve_ok:
        tl_old = np.ones(n)
        tl_old[basal_or_isolate_mask] = 1
        max_iter = 5000
        tol = 1e-9
        converged = False
        tl_new = tl_old
        for _ in range(max_iter):
            tl_new = 1 + P.dot(tl_old)
            tl_new[basal_or_isolate_mask] = 1
            if np.any(~np.isfinite(tl_new)) or np.any(np.abs(tl_new) > 1e6):
                converged = False
                break
            if np.max(np.abs(tl_new - tl_old)) < tol:
                converged = True
                break
            tl_old = tl_new

        if converged and np.all(tl_new >= 1 - 1e-8) and np.all(tl_new <= max(20, n)):
            trophic_level = tl_new
        else:
            trophic_level = np.full(n, np.nan)
            trophic_level[basal_or_isolate_mask] = 1

    finite_tl = trophic_level[np.isfinite(trophic_level)]
    mean_trophic_level = float(np.mean(finite_tl)) if finite_tl.size else np.nan

    return {
        'NumSpecies': n,
        'NumLinks': links,
        'Connectance': connectance,
        'MeanTrophicLevel': mean_trophic_level,
        'MeanDegree': mean_degree,
        'MeanGenerality': mean_generality,
        'MeanVulnerability': mean_vulnerability,
        'PropBasal': safe_divide(int(np.sum(basal_mask)), n),
        'PropIntermediate': safe_divide(int(np.sum(intermediate_mask)), n),
        'PropTop': safe_divide(int(np.sum(top_mask)), n),
    }


def compare_empirical_pseudo_webs(empirical_full, pseudo_full):
    empirical_full = binarize_adjacency(empirical_full)
    pseudo_full = binarize_adjacency(pseudo_full)
    n = empirical_full.shape[0]

    empirical_links = int(empirical_full.nnz)
    pseudo_links = int(pseudo_full.nnz)
    tp = int(empirical_full.multiply(pseudo_full).nnz)
    fp = pseudo_links - tp
    fn = empirical_links - tp
    tn = n * max(0, n - 1) - tp - fp - fn

    tpr = safe_divide(tp, tp + fn)
    tnr = safe_divide(tn, tn + fp)
    fpr = safe_divide(fp, fp + tn)
    fnr = safe_divide(fn, fn + tp)
    precision = safe_divide(tp, tp + fp)
    recall = tpr
    f1_score = safe_divide(2 * precision * recall, precision + recall)
    jaccard_links = safe_divide(tp, tp + fp + fn)

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'TPR': tpr,
        'TNR': tnr,
        'FPR': fpr,
        'FNR': fnr,
        'Precision': precision,
        'Recall': recall,
        'F1Score': f1_score,
        'TSS': tpr + tnr - 1,
        'JaccardLinks': jaccard_links,
        'EmpiricalLinks': empirical_links,
        'PseudoLinks': pseudo_links,
        'LinkDelta': pseudo_links - empirical_links,
    }


def pair_tuple_to_array(pairs):
    if pairs is None:
        return np.empty((0, 2), dtype=int)
    return np.column_stack([np.asarray(pairs[0], dtype=int), np.asarray(pairs[1], dtype=int)])


def build_ecological_metric_row(net, observed_train, test_pos, test_neg, labels, scores, threshold):
    n = net.shape[0]
    empirical_full = binarize_adjacency(net)

    test_pairs = np.vstack([pair_tuple_to_array(test_pos), pair_tuple_to_array(test_neg)])
    predictions = np.asarray(scores) > threshold
    predicted_links = test_pairs[predictions] if test_pairs.size else np.empty((0, 2), dtype=int)

    if predicted_links.size:
        predicted_sparse = ssp.csr_matrix(
            (np.ones(predicted_links.shape[0]), (predicted_links[:, 0], predicted_links[:, 1])),
            shape=(n, n),
        )
    else:
        predicted_sparse = ssp.csr_matrix((n, n))

    pseudo_full = binarize_adjacency(observed_train + predicted_sparse)
    empirical_metrics = compute_foodweb_metrics(empirical_full)
    pseudo_metrics = compute_foodweb_metrics(pseudo_full)
    comparison_metrics = compare_empirical_pseudo_webs(empirical_full, pseudo_full)

    row = {
        'EmpiricalNumSpecies': empirical_metrics['NumSpecies'],
        'EmpiricalLinks': empirical_metrics['NumLinks'],
        'EmpiricalConnectance': empirical_metrics['Connectance'],
        'EmpiricalMeanTrophicLevel': empirical_metrics['MeanTrophicLevel'],
        'EmpiricalMeanDegree': empirical_metrics['MeanDegree'],
        'EmpiricalMeanGenerality': empirical_metrics['MeanGenerality'],
        'EmpiricalMeanVulnerability': empirical_metrics['MeanVulnerability'],
        'EmpiricalPropBasal': empirical_metrics['PropBasal'],
        'EmpiricalPropIntermediate': empirical_metrics['PropIntermediate'],
        'EmpiricalPropTop': empirical_metrics['PropTop'],
        'PseudoNumSpecies': pseudo_metrics['NumSpecies'],
        'PseudoLinks': pseudo_metrics['NumLinks'],
        'PseudoConnectance': pseudo_metrics['Connectance'],
        'PseudoMeanTrophicLevel': pseudo_metrics['MeanTrophicLevel'],
        'PseudoMeanDegree': pseudo_metrics['MeanDegree'],
        'PseudoMeanGenerality': pseudo_metrics['MeanGenerality'],
        'PseudoMeanVulnerability': pseudo_metrics['MeanVulnerability'],
        'PseudoPropBasal': pseudo_metrics['PropBasal'],
        'PseudoPropIntermediate': pseudo_metrics['PropIntermediate'],
        'PseudoPropTop': pseudo_metrics['PropTop'],
        'DeltaLinks': pseudo_metrics['NumLinks'] - empirical_metrics['NumLinks'],
        'DeltaConnectance': pseudo_metrics['Connectance'] - empirical_metrics['Connectance'],
        'DeltaMeanTrophicLevel': pseudo_metrics['MeanTrophicLevel'] - empirical_metrics['MeanTrophicLevel'],
        'DeltaMeanDegree': pseudo_metrics['MeanDegree'] - empirical_metrics['MeanDegree'],
        'DeltaMeanGenerality': pseudo_metrics['MeanGenerality'] - empirical_metrics['MeanGenerality'],
        'DeltaMeanVulnerability': pseudo_metrics['MeanVulnerability'] - empirical_metrics['MeanVulnerability'],
        'DeltaPropBasal': pseudo_metrics['PropBasal'] - empirical_metrics['PropBasal'],
        'DeltaPropIntermediate': pseudo_metrics['PropIntermediate'] - empirical_metrics['PropIntermediate'],
        'DeltaPropTop': pseudo_metrics['PropTop'] - empirical_metrics['PropTop'],
        'PseudoTP': comparison_metrics['TP'],
        'PseudoFP': comparison_metrics['FP'],
        'PseudoFN': comparison_metrics['FN'],
        'PseudoTN': comparison_metrics['TN'],
        'PseudoTPR': comparison_metrics['TPR'],
        'PseudoTNR': comparison_metrics['TNR'],
        'PseudoFPR': comparison_metrics['FPR'],
        'PseudoFNR': comparison_metrics['FNR'],
        'PseudoPrecision': comparison_metrics['Precision'],
        'PseudoRecall': comparison_metrics['Recall'],
        'PseudoF1Score': comparison_metrics['F1Score'],
        'PseudoTSS': comparison_metrics['TSS'],
        'PseudoJaccardLinks': comparison_metrics['JaccardLinks'],
        'NumPredictedNovelLinks': int(predicted_links.shape[0]),
        'NumTrueNovelLinks': int(np.sum(np.asarray(labels) == 1)),
        'EvaluateOnAllUnseen': 0,
    }
    return row


def predict_graph_scores(classifier, graphs, batch_size):
    classifier.eval()
    scores = []
    labels = []
    batch_graph = []

    with torch.no_grad():
        for i, graph in enumerate(graphs):
            batch_graph.append(graph)
            if len(batch_graph) == batch_size or i == (len(graphs) - 1):
                logits = classifier(batch_graph)[0]
                scores.append(logits[:, 1].exp().cpu().detach().numpy())
                labels.extend([g.label for g in batch_graph])
                batch_graph = []

    if scores:
        scores = np.concatenate(scores, axis=0)
    else:
        scores = np.asarray([], dtype=np.float64)

    return np.asarray(labels, dtype=np.int64), scores


def safe_file_stem(value):
    value = os.path.splitext(os.path.basename(str(value or 'seal')))[0]
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in value)


def resolve_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.exists(cwd_path):
        return cwd_path
    return os.path.abspath(os.path.join(base_dir, path))


def format_elapsed_time(elapsed_seconds):
    elapsed_seconds = int(round(elapsed_seconds))
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


def write_metrics_csv(row, result_dir, data_name):
    os.makedirs(result_dir, exist_ok=True)

    fieldnames = [
        'Iteration',
        'Version',
        'ROC_AUC',
        'PR_AUC',
        'TimeElapsed',
        'K',
        'TrainRatio',
        'ExperimentID',
        'Seed',
        'ThresholdMode',
        'Threshold',
        'Precision',
        'Recall',
        'F1Score',
        'TotalLinks',
        'TrainLinks',
        'TestLinks',
        'BackboneTotal',
        'NonBackboneTotal',
        'BackboneTrainLinks',
        'NonBackboneTrainLinks',
        'BackboneTestLinks',
        'NonBackboneTestLinks',
        'EmpiricalNumSpecies',
        'EmpiricalLinks',
        'EmpiricalConnectance',
        'EmpiricalMeanTrophicLevel',
        'EmpiricalMeanDegree',
        'EmpiricalMeanGenerality',
        'EmpiricalMeanVulnerability',
        'EmpiricalPropBasal',
        'EmpiricalPropIntermediate',
        'EmpiricalPropTop',
        'PseudoNumSpecies',
        'PseudoLinks',
        'PseudoConnectance',
        'PseudoMeanTrophicLevel',
        'PseudoMeanDegree',
        'PseudoMeanGenerality',
        'PseudoMeanVulnerability',
        'PseudoPropBasal',
        'PseudoPropIntermediate',
        'PseudoPropTop',
        'DeltaLinks',
        'DeltaConnectance',
        'DeltaMeanTrophicLevel',
        'DeltaMeanDegree',
        'DeltaMeanGenerality',
        'DeltaMeanVulnerability',
        'DeltaPropBasal',
        'DeltaPropIntermediate',
        'DeltaPropTop',
        'PseudoTP',
        'PseudoFP',
        'PseudoFN',
        'PseudoTN',
        'PseudoTPR',
        'PseudoTNR',
        'PseudoFPR',
        'PseudoFNR',
        'PseudoPrecision',
        'PseudoRecall',
        'PseudoF1Score',
        'PseudoTSS',
        'PseudoJaccardLinks',
        'NumPredictedNovelLinks',
        'NumTrueNovelLinks',
        'EvaluateOnAllUnseen',
        'CvK',
        'FoldID',
        'NumFolds',
    ]

    result_file = os.path.join(
        result_dir,
        '{}_results_SEAL.csv'.format(safe_file_stem(data_name))
    )

    write_header = not os.path.isfile(result_file) or os.path.getsize(result_file) == 0
    iteration = 1
    if not write_header:
        with open(result_file, newline='') as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
            if existing_header != fieldnames:
                raise ValueError(
                    'Existing result CSV has an incompatible header: {}. '
                    'Use --overwrite-results in run_foodweb_seal.py or remove the old file.'.format(
                        result_file
                    )
                )
            iteration = sum(1 for _ in reader) + 1

    row = dict(row)
    row['Iteration'] = iteration
    integer_fields = {
        'Iteration',
        'K',
        'ExperimentID',
        'Seed',
        'TotalLinks',
        'TrainLinks',
        'TestLinks',
        'BackboneTotal',
        'NonBackboneTotal',
        'BackboneTrainLinks',
        'NonBackboneTrainLinks',
        'BackboneTestLinks',
        'NonBackboneTestLinks',
        'EmpiricalNumSpecies',
        'EmpiricalLinks',
        'PseudoNumSpecies',
        'PseudoLinks',
        'DeltaLinks',
        'PseudoTP',
        'PseudoFP',
        'PseudoFN',
        'PseudoTN',
        'NumPredictedNovelLinks',
        'NumTrueNovelLinks',
        'EvaluateOnAllUnseen',
        'CvK',
        'FoldID',
        'NumFolds',
    }
    for key in fieldnames:
        value = row.get(key)
        if value != '' and value is not None:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if np.isnan(numeric_value):
                row[key] = 'NaN'
            elif key in integer_fields:
                row[key] = int(round(numeric_value))
            else:
                row[key] = '{:.4f}'.format(numeric_value)

    with open(result_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, '') for name in fieldnames})

    return result_file


parser = argparse.ArgumentParser(description='Link Prediction with SEAL')
# general settings
parser.add_argument('--data-name', default=None, help='network name')
parser.add_argument('--mat-folder', default=None,
                    help='folder containing <data-name>.mat; defaults to ./data')
parser.add_argument('--result-dir', default=None,
                    help='directory for SEAL result CSVs; defaults to ./data/result')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--only-predict', action='store_true', default=False,
                    help='if True, will load the saved model and output predictions\
                    for links in test-name; you still need to specify train-name\
                    in order to build the observed network and extract subgraphs')
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--max-train-num', type=int, default=100000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--experiment-id', type=int, default=1,
                    help='experiment id written to the result CSV')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--num-epochs', type=int, default=50,
                    help='number of SEAL/DGCNN training epochs')
parser.add_argument('--threshold', type=float, default=0.5,
                    help='fixed threshold for precision/recall/F1')
parser.add_argument('--no-parallel', action='store_true', default=False,
                    help='if True, use single thread for subgraph extraction; \
                    by default use all cpu cores to extract subgraphs in parallel')
parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                    help='if True, regard all unknown links as negative test data; \
                    sample a portion from them as negative training data. Otherwise,\
                    train negative and test negative data are both sampled from \
                    unknown links without overlap.')
# model settings
parser.add_argument('--hop', default=1, metavar='S', 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=None, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,
                    help='whether to use node2vec node embeddings')
parser.add_argument('--use-attribute', action='store_true', default=False,
                    help='whether to use node attributes')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='save the final model')
args = parser.parse_args()
run_start_time = time.time()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

cmd_args.seed = args.seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)


'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath(__file__))

# check whether train and test links are provided
train_pos, test_pos = None, None
if args.train_name is not None:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    train_pos = (train_idx[:, 0], train_idx[:, 1])
if args.test_name is not None:
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    test_pos = (test_idx[:, 0], test_idx[:, 1])

# build observed network
if args.data_name is not None:  # use .mat network
    mat_folder = resolve_path(args.mat_folder, args.file_dir)
    if mat_folder is None:
        mat_folder = os.path.join(args.file_dir, 'data')
    args.data_dir = os.path.join(mat_folder, '{}.mat'.format(args.data_name))
    data = sio.loadmat(args.data_dir)
    net = data['net']
    if 'group' in data:
        # load node attributes (here a.k.a. node classes)
        attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
else:  # build network from train links
    assert (args.train_name is not None), "must provide train links if not using .mat"
    if args.train_name.endswith('_train.txt'):
        args.data_name = args.train_name[:-10] 
    else:
        args.data_name = args.train_name.split('.')[0]
    max_idx = np.max(train_idx)
    if args.test_name is not None:
        max_idx = max(max_idx, np.max(test_idx))
    net = ssp.csc_matrix(
        (np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), 
        shape=(max_idx+1, max_idx+1)
    )
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops

# sample train and test links
if args.train_name is None and args.test_name is None:
    # sample both positive and negative train/test links from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, args.test_ratio, max_train_num=args.max_train_num
    )
else:
    # use provided train/test positive links, sample negative from net
    train_pos, train_neg, test_pos, test_neg = sample_neg(
        net, 
        train_pos=train_pos, 
        test_pos=test_pos, 
        max_train_num=args.max_train_num,
        all_unknown_as_negative=args.all_unknown_as_negative
    )

'''Train and apply classifier'''
A = net.copy()  # the observed network
A[test_pos[0], test_pos[1]] = 0  # mask test links
A[test_pos[1], test_pos[0]] = 0  # mask test links
A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x

node_information = None
if args.use_embedding:
    embeddings = generate_node2vec_embeddings(A, 128, True, train_neg)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

if args.only_predict:  # no need to use negatives
    _, test_graphs, max_n_label = links2subgraphs(
        A, 
        None, 
        None, 
        test_pos, # test_pos is a name only, we don't actually know their labels
        None, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel
    )
    print('# test: %d' % (len(test_graphs)))
else:
    train_graphs, test_graphs, max_n_label = links2subgraphs(
        A, 
        train_pos, 
        train_neg, 
        test_pos, 
        test_neg, 
        args.hop, 
        args.max_nodes_per_hop, 
        node_information, 
        args.no_parallel
    )
    print('# train: %d, # test: %d' % (len(train_graphs), len(test_graphs)))

# DGCNN configurations
if args.only_predict:
    with open('data/{}_hyper.pkl'.format(args.data_name), 'rb') as hyperparameters_name:
        saved_cmd_args = pickle.load(hyperparameters_name)
    for key, value in vars(saved_cmd_args).items(): # replace with saved cmd_args
        vars(cmd_args)[key] = value
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    model_name = 'data/{}_model.pth'.format(args.data_name)
    classifier.load_state_dict(torch.load(model_name))
    classifier.eval()
    predictions = []
    batch_graph = []
    for i, graph in enumerate(test_graphs):
        batch_graph.append(graph)
        if len(batch_graph) == cmd_args.batch_size or i == (len(test_graphs)-1):
            predictions.append(classifier(batch_graph)[0][:, 1].exp().cpu().detach())
            batch_graph = []
    predictions = torch.cat(predictions, 0).unsqueeze(1).numpy()
    test_idx_and_pred = np.concatenate([test_idx, predictions], 1)
    pred_name = 'data/' + args.test_name.split('.')[0] + '_pred.txt'
    np.savetxt(pred_name, test_idx_and_pred, fmt=['%d', '%d', '%1.2f'])
    print('Predictions for {} are saved in {}'.format(args.test_name, pred_name))
    exit()


cmd_args.gm = 'DGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu' if args.cuda else 'cpu'
cmd_args.num_epochs = args.num_epochs
cmd_args.learning_rate = 1e-4
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    k_ = int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1
    cmd_args.sortpooling_k = max(10, num_nodes_list[k_])
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))

classifier = Classifier()
if cmd_args.mode == 'gpu':
    classifier = classifier.cuda()

optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

random.shuffle(train_graphs)
val_num = max(1, int(0.1 * len(train_graphs))) if len(train_graphs) > 1 else 0
val_graphs = train_graphs[:val_num]
train_graphs = train_graphs[val_num:]
loop_batch_size = min(args.batch_size, max(1, len(train_graphs)))

train_idxes = list(range(len(train_graphs)))
best_loss = None
best_epoch = None
best_test_metrics = None
best_val_loss = None
best_test_loss = None
for epoch in range(cmd_args.num_epochs):
    random.shuffle(train_idxes)
    classifier.train()
    avg_loss = loop_dataset(
        train_graphs, classifier, train_idxes, optimizer=optimizer, bsize=loop_batch_size
    )
    if not cmd_args.printAUC:
        avg_loss[2] = 0.0
    print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, avg_loss[0], avg_loss[1], avg_loss[2]))

    classifier.eval()
    val_loss = loop_dataset(
        val_graphs, classifier, list(range(len(val_graphs))), bsize=loop_batch_size
    )
    if not cmd_args.printAUC:
        val_loss[2] = 0.0
    print('\033[93maverage validation of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
        epoch, val_loss[0], val_loss[1], val_loss[2]))
    if best_loss is None:
        best_loss = val_loss
    if val_loss[0] <= best_loss[0]:
        best_loss = val_loss
        best_epoch = epoch
        test_loss = loop_dataset(
            test_graphs, classifier, list(range(len(test_graphs))), bsize=loop_batch_size
        )
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[94maverage test of epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
            epoch, test_loss[0], test_loss[1], test_loss[2]))
        test_labels, test_scores = predict_graph_scores(classifier, test_graphs, loop_batch_size)
        best_test_metrics = compute_link_prediction_metrics(test_labels, test_scores, threshold=args.threshold)
        best_val_loss = val_loss.copy()
        best_test_loss = test_loss.copy()

print('\033[95mFinal test performance: epoch %d: loss %.5f acc %.5f auc %.5f\033[0m' % (
    best_epoch, test_loss[0], test_loss[1], test_loss[2]))

if best_test_metrics is None:
    test_labels, test_scores = predict_graph_scores(classifier, test_graphs, loop_batch_size)
    best_test_metrics = compute_link_prediction_metrics(test_labels, test_scores, threshold=args.threshold)
    best_val_loss = val_loss.copy()
    best_test_loss = test_loss.copy()

result_dir = resolve_path(args.result_dir, args.file_dir)
if result_dir is None:
    result_dir = os.path.join(args.file_dir, 'data', 'result', 'prediction_scores_logs')
train_ratio = safe_divide(len(train_pos[0]), len(train_pos[0]) + len(test_pos[0])) * 100
total_links = int(binarize_adjacency(net).nnz)
train_links = len(train_pos[0])
test_links = len(test_pos[0])
ecological_row = build_ecological_metric_row(
    net,
    A,
    test_pos,
    test_neg,
    test_labels,
    test_scores,
    args.threshold,
)
metrics_row = {
    'Version': 'SEAL_undirected',
    'TimeElapsed': format_elapsed_time(time.time() - run_start_time),
    'K': args.hop,
    'TrainRatio': train_ratio,
    'ExperimentID': args.experiment_id,
    'Seed': args.seed,
    'ThresholdMode': 'fixed',
    'TotalLinks': total_links,
    'TrainLinks': train_links,
    'TestLinks': test_links,
    'BackboneTotal': 0,
    'NonBackboneTotal': total_links,
    'BackboneTrainLinks': 0,
    'NonBackboneTrainLinks': train_links,
    'BackboneTestLinks': 0,
    'NonBackboneTestLinks': test_links,
    'CvK': 0,
    'FoldID': 0,
    'NumFolds': 0,
}
metrics_row.update(best_test_metrics)
metrics_row.update(ecological_row)
metrics_file = write_metrics_csv(metrics_row, result_dir, args.data_name)
print('SEAL metrics appended to {}'.format(metrics_file))
        
if args.save_model:
    model_name = 'data/{}_model.pth'.format(args.data_name)
    print('Saving final model states to {}...'.format(model_name))
    torch.save(classifier.state_dict(), model_name)
    hyper_name = 'data/{}_hyper.pkl'.format(args.data_name)
    with open(hyper_name, 'wb') as hyperparameters_file:
        pickle.dump(cmd_args, hyperparameters_file)
        print('Saving hyperparameters to {}...'.format(hyper_name))
