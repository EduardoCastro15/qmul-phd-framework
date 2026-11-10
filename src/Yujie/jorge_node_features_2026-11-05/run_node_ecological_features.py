"""Jorge local subgraphs with aligned per-node ecology, paired to frozen baseline."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from run_experiment import HERE,SOURCE,FOLDS,sha,write_csv,measure
from run_fold_union_expanded_ecology import prepare
from run_fold_union_training_fraction_sweep_10_80 import OUTPUT as BASELINE,PERCENTAGES,split_fraction
from run_ecological_attribute_comparison import OUTPUT as ATTRIBUTE_SOURCE,DATRAS,MEMBERS
from ecological_attributes import role
from node_ecological_encoder import ordered_subgraphs,fit_node_preprocessor,transform_nodes,aligned_node_block
from wlnm import Encoder,MLP

OUTPUT=HERE.parent/'output/jorge_aligned_node_ecology_train10_80_2002_2005_2008_2026-10-22'


def prepare_nodes(jobs):
    cfg=json.loads((ATTRIBUTE_SOURCE/'PROTOCOL.json').read_text())
    assert sha(DATRAS)==cfg['datras_sha256'] and sha(MEMBERS)==cfg['membership_sha256']
    OUTPUT.joinpath('node_attributes').mkdir(exist_ok=True)
    source_hashes={};coverage=[]
    for job in jobs:
        path=ATTRIBUTE_SOURCE/'attributes'/f"{job['year']}_Fold{job['fold']}_source_values.csv"
        d=pd.read_csv(path);source_hashes[str(path)]=sha(path)
        numeric=[];rows=[]
        for name in job['node_names']:
            q=d[d.prey==name]
            assert len(q)>0
            vals=[]
            for col in ['resource_biomass','resource_individual_weight_g']:
                unique=q[col].dropna().unique();assert len(unique)<=1
                value=float(unique[0]) if len(unique) else np.nan;vals.append(value)
            # Existing attributes are independently surveyed node values and
            # must not depend on which edge references that node.
            other=d[d.predator==name]
            for val,col in zip(vals,['consumer_biomass','consumer_individual_weight_g']):
                u=other[col].dropna().unique();assert len(u)<=1
                if len(u):np.testing.assert_allclose(val,float(u[0]),equal_nan=True)
            biomass,mass=vals
            numeric.append([np.log1p(biomass) if np.isfinite(biomass) and biomass>=0 else np.nan,
                            np.log(mass) if np.isfinite(mass) and mass>0 else np.nan])
            rows.append(dict(node=name,biomass_density=biomass,individual_weight_g=mass,fixed_feeding_role=role(name)))
        job['node_numeric']=np.asarray(numeric,float)
        job['node_roles']=np.asarray([role(n) for n in job['node_names']],str)
        pd.DataFrame(rows).to_csv(OUTPUT/'node_attributes'/f"{job['year']}_Fold{job['fold']}.csv",index=False)
        coverage.append(dict(year=job['year'],fold=job['fold'],nodes=286,biomass_available=int(np.isfinite(job['node_numeric'][:,0]).sum()),
                             individual_weight_available=int(np.isfinite(job['node_numeric'][:,1]).sum()),role_known=int((job['node_roles']!='unknown').sum())))
    pd.DataFrame(coverage).to_csv(OUTPUT/'node_attribute_coverage.csv',index=False)
    return jobs,source_hashes


def run_job(job):
    percent=job['percent'];out=OUTPUT/f'train_{percent:02}'/str(job['year'])/job['group_id'];out.mkdir(parents=True,exist_ok=True)
    done=out/'runs.json'
    if done.exists():return json.loads(done.read_text())
    ref=BASELINE/f'train_{percent:02}'/'networks'/str(job['year'])/job['group_id']
    labels=job['extended_labels'];pairs=job['extended_pairs'];synthetic=job['synthetic']
    rows=[];scores=[];ordered=[];states=[];weights={}
    with np.load(ref/'predictions.npz') as old:
        np.testing.assert_array_equal(labels,old['labels']);np.testing.assert_array_equal(pairs,old['candidate_pairs'])
        for i,seed in enumerate(old['seeds']):
            tr=old['train_candidate_indices'][i];te=old['test_candidate_indices'][i]
            expected=split_fraction(labels,synthetic,int(seed),percent)
            np.testing.assert_array_equal(tr,expected[0]);np.testing.assert_array_equal(te,expected[1])
            graph=np.zeros((286,286),np.uint8);positive=pairs[tr[labels[tr]==1]];graph[positive[:,0],positive[:,1]]=1
            assert not graph[pairs[te,0],pairs[te,1]].any()
            topology,node_ids=ordered_subgraphs(Encoder(graph),pairs[np.r_[tr,te]])
            training_nodes=np.unique(node_ids[:len(tr)][node_ids[:len(tr)]>=0])
            state=fit_node_preprocessor(job['node_numeric'],job['node_roles'],training_nodes)
            features=transform_nodes(job['node_numeric'],job['node_roles'],state)
            aligned=aligned_node_block(node_ids,features)
            x=np.column_stack([topology,aligned])
            assert np.isfinite(x).all()
            model=MLP(x.shape[1],int(seed)+1000000)
            loss=model.fit(x[:len(tr)],labels[tr].astype(int),int(seed)+2000000)
            score=model.predict(x[len(tr):]);scores.append(score);ordered.append(node_ids);states.append(state)
            for j,w in enumerate(model.weights):weights[f'r{i+1:03}_weight_{j}']=w
            for j,b in enumerate(model.biases):weights[f'r{i+1:03}_bias_{j}']=b
            for name,prob in [('jorge',old['probabilities'][i]),('node_ecology',score)]:
                rec=dict(model=name,train_percent=percent,year=job['year'],fold=job['fold'],repeat=i+1,seed=int(seed),
                    feature_dim=90 if name=='jorge' else int(x.shape[1]),n_train=len(tr),n_train_positive=int((labels[tr]==1).sum()),
                    n_train_negative=int((labels[tr]==0).sum()),n_test_synthetic=int(synthetic[te].sum()),
                    actual_positive_train_fraction=float((labels[tr]==1).sum()/job['n_positive']),train_loss=None if name=='jorge' else loss)
                for stratum,mask in [('all',np.ones(len(te),bool)),('known',~synthetic[te])]:
                    rec.update({stratum+'_'+k:v for k,v in measure(labels[te][mask],prob[mask]).items()})
                rows.append(rec)
        np.savez_compressed(out/'predictions.npz',node_ecology=np.stack(scores),jorge=old['probabilities'],
            train_candidate_indices=old['train_candidate_indices'],test_candidate_indices=old['test_candidate_indices'],
            seeds=old['seeds'],canonical_node_ids=np.stack(ordered),baseline_reference=str(ref))
    np.savez_compressed(out/'models.npz',**weights)
    (out/'node_preprocessors.json').write_text(json.dumps(states,indent=2)+'\n')
    write_csv(out/'runs.csv',rows);done.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    print(f"DONE {percent}% {job['year']}/Fold{job['fold']}: 50 aligned-node fits",flush=True)
    return rows


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest,jobs=prepare(OUTPUT,50);jobs,source_hashes=prepare_nodes(jobs)
    cfg=dict(years=[2002,2005,2008],train_percentages=PERCENTAGES,repeats=50,models=['jorge','node_ecology'],
        baseline=str(BASELINE),negative_ratio=2,k=10,
        classifier='Original MLP hidden layers 32,32,16,2; SGDM, 200 epochs; only first-layer input dimension changes',
        topology='Original Encoder.encode(return_debug=True); exact original 90-dimensional vector; no change to extraction, masking or canonicalization',
        node_alignment='ordered IDs = debug.nodes[debug.order]; pad node slots to K=10 with ID -1; node-major flattening',
        node_attributes=['log1p survey biomass density','log survey mean individual weight','two missing indicators','training-only one-hot feeding role','real-node padding mask'],
        preprocessing='Fit medians, center/SD and categories on unique nodes present in TRAINING candidate subgraphs only; test nodes only transformed',
        disallowed_attributes=['pair ratios','pair identity','synthetic flag','test labels','full-network trophic levels','other-web edge frequencies','FinFW constructed M/N'],
        data_policy='Same-year DATRAS before-interpolation data, previously frozen surveyed node values and fixed conservative roles',
        attribute_source_values_hashes=source_hashes,
        source_sha256=sha(SOURCE),fold_sha256=sha(FOLDS),datras_sha256=sha(DATRAS),membership_sha256=sha(MEMBERS),
        code_hashes={n:sha(HERE/n) for n in ['run_node_ecological_features.py','node_ecological_encoder.py','ecological_attributes.py','wlnm.py','run_experiment.py']},
        aggregation='Equal-weight five-fold metric per repeat; mean +/- sample STD across 50 repeats; canonical-only subset separately')
    path=OUTPUT/'PROTOCOL.json'
    if path.exists():assert json.loads(path.read_text())==cfg
    path.write_text(json.dumps(cfg,indent=2)+'\n')
    records=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run_job,dict(job,percent=p)) for job in jobs for p in PERCENTAGES]
        for future in as_completed(futures):records.extend(future.result())
    records.sort(key=lambda r:(r['model'],r['year'],r['train_percent'],r['fold'],r['repeat']))
    assert len(records)==12000;write_csv(OUTPUT/'runs.csv',records)
    for p,key in [(SOURCE,'source_sha256'),(FOLDS,'fold_sha256'),(DATRAS,'datras_sha256'),(MEMBERS,'membership_sha256')]:assert sha(p)==cfg[key]
    (OUTPUT/'TRAINING_COMPLETION.json').write_text(json.dumps(dict(results=12000,new_node_feature_fits=6000,reused_jorge_fits=6000),indent=2)+'\n')
    print('Training complete: 6000 new aligned-node fits; 12000 paired results',flush=True)


if __name__=='__main__':main()
