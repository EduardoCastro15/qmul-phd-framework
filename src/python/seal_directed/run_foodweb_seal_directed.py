import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

# conda activate Foodweb
# python src/python/seal_directed/run_foodweb_seal_directed.py \
#   --foodweb-csv src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv \
#   --mat-folder src/python/seal_directed/data/foodwebs_mat_seal_attrs \
#   --train-ratio 0.9 \
#   --num-experiments 1 \
#   --hop 1 \
#   --no-parallel

# ---


# Comando para correr SEAL dirigido con atributos:

# python src/python/seal_directed/run_foodweb_seal_directed.py \
#   --foodweb-csv src/matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv \
#   --train-ratio 0.9 \
#   --num-experiments 3 \
#   --hop 1 \
#   --base-seed 12345 \
#   --use-attribute \
#   --continue-on-error \
#   --no-parallel


def safe_file_stem(value):
    value = os.path.splitext(os.path.basename(str(value or 'seal')))[0]
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in value)


def read_foodweb_names(foodweb_csv):
    with open(foodweb_csv, newline='') as f:
        reader = csv.DictReader(f)
        if 'Foodweb' not in reader.fieldnames:
            raise ValueError('CSV must contain a Foodweb column: {}'.format(foodweb_csv))
        return [row['Foodweb'].strip() for row in reader if row.get('Foodweb', '').strip()]


def build_command(args, script_dir, foodweb, seed, experiment_id, test_ratio):
    cmd = [
        str(args.python_executable),
        str(script_dir / 'Main_directed.py'),
        '--data-name',
        foodweb,
        '--mat-folder',
        str(args.mat_folder),
        '--result-dir',
        str(args.result_dir),
        '--hop',
        str(args.hop),
        '--test-ratio',
        str(test_ratio),
        '--seed',
        str(seed),
        '--experiment-id',
        str(experiment_id),
        '--batch-size',
        str(args.batch_size),
        '--max-train-num',
        str(args.max_train_num),
        '--num-epochs',
        str(args.num_epochs),
        '--threshold',
        str(args.threshold),
    ]

    if not args.cuda:
        cmd.append('--no-cuda')
    if args.no_parallel:
        cmd.append('--no-parallel')
    if args.all_unknown_as_negative:
        cmd.append('--all-unknown-as-negative')
    if args.no_role_filter:
        cmd.append('--no-role-filter')
    if args.max_nodes_per_hop is not None:
        cmd.extend(['--max-nodes-per-hop', str(args.max_nodes_per_hop)])
    if args.use_embedding:
        cmd.append('--use-embedding')
    if args.use_attribute:
        cmd.append('--use-attribute')

    return cmd


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_foodweb_csv = script_dir / '../../matlab/data/foodwebs_mat/foodweb_metrics_ecosystem.csv'
    default_mat_folder = script_dir / 'data/foodwebs_mat_seal_attrs'
    default_result_dir = script_dir / 'data/result/prediction_scores_logs'
    default_terminal_log_dir = script_dir / 'data/result/terminal_logs'

    parser = argparse.ArgumentParser(
        description='Run directed SEAL over food webs listed in a CSV file.'
    )
    parser.add_argument('--foodweb-csv', type=Path, default=default_foodweb_csv,
                        help='CSV containing a Foodweb column')
    parser.add_argument('--mat-folder', type=Path, default=default_mat_folder,
                        help='folder containing <Foodweb>.mat files')
    parser.add_argument('--result-dir', type=Path, default=default_result_dir,
                        help='directory where result CSVs are written')
    parser.add_argument('--terminal-log-dir', type=Path, default=default_terminal_log_dir,
                        help='directory where terminal logs are written')
    parser.add_argument('--python-executable', type=Path, default=Path(sys.executable),
                        help='Python executable used to launch Main_directed.py')
    parser.add_argument('--hop', default=1,
                        help='SEAL enclosing subgraph hop number, e.g. 1, 2, or auto')
    parser.add_argument('--train-ratio', type=float, default=0.9,
                        help='fraction of positive links used for training')
    parser.add_argument('--num-experiments', type=int, default=5,
                        help='number of repeated SEAL runs per food web')
    parser.add_argument('--only-foodweb', action='append', default=[],
                        help='run only this food-web name; can be provided multiple times')
    parser.add_argument('--limit', type=int, default=None,
                        help='run only the first N food webs after filtering')
    parser.add_argument('--base-seed', type=int, default=12345,
                        help='first seed; experiment i uses base_seed + i - 1')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--max-train-num', type=int, default=100000)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max-nodes-per-hop', type=int, default=None)
    parser.add_argument('--cuda', action='store_true',
                        help='allow CUDA if torch can use it; default is CPU')
    parser.add_argument('--no-parallel', action='store_true',
                        help='disable multiprocessing subgraph extraction')
    parser.add_argument('--all-unknown-as-negative', action='store_true')
    parser.add_argument('--no-role-filter', action='store_true',
                        help='disable directed ecological role filtering for negative links')
    parser.add_argument('--use-embedding', action='store_true')
    parser.add_argument('--use-attribute', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='print commands without running them')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='keep running remaining food webs after a failed run')
    parser.add_argument('--overwrite-results', action='store_true',
                        help='remove existing <Foodweb>_results_SEAL_directed.csv files before running')
    return parser.parse_args()


def validate_python_environment(python_executable):
    required_modules = [
        'torch',
        'numpy',
        'scipy',
        'sklearn',
        'networkx',
        'gensim',
    ]
    code = 'import ' + ', '.join(required_modules)
    completed = subprocess.run(
        [str(python_executable), '-c', code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            'Python executable "{}" cannot import the SEAL dependencies. '
            'Run from the Foodweb conda environment or pass '
            '--python-executable /Users/acw792/miniconda3/envs/Foodweb/bin/python. '
            'Original error:\n{}'.format(
                python_executable,
                completed.stderr.strip() or completed.stdout.strip(),
            )
        )


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    args.foodweb_csv = args.foodweb_csv.resolve()
    args.mat_folder = args.mat_folder.resolve()
    args.result_dir = args.result_dir.resolve()
    args.terminal_log_dir = args.terminal_log_dir.resolve()
    args.python_executable = args.python_executable.resolve()

    if not args.foodweb_csv.is_file():
        raise FileNotFoundError(args.foodweb_csv)
    if not args.mat_folder.is_dir():
        raise FileNotFoundError(args.mat_folder)
    if not args.python_executable.is_file():
        raise FileNotFoundError(args.python_executable)
    if not 0 < args.train_ratio < 1:
        raise ValueError('--train-ratio must be between 0 and 1')
    if not args.dry_run:
        validate_python_environment(args.python_executable)

    foodwebs = read_foodweb_names(args.foodweb_csv)
    if args.only_foodweb:
        requested = set(args.only_foodweb)
        foodwebs = [foodweb for foodweb in foodwebs if foodweb in requested]
        missing = requested.difference(foodwebs)
        if missing:
            raise ValueError('Requested food webs not found in CSV: {}'.format(
                ', '.join(sorted(missing))
            ))
    if args.limit is not None:
        foodwebs = foodwebs[:args.limit]

    if not foodwebs:
        raise ValueError('No food webs selected to run.')
    args.result_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.terminal_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite_results:
        for foodweb in foodwebs:
            result_file = args.result_dir / '{}_results_SEAL_directed.csv'.format(
                safe_file_stem(foodweb)
            )
            if result_file.exists():
                result_file.unlink()

    test_ratio = 1.0 - args.train_ratio
    failures = []

    for foodweb in foodwebs:
        mat_file = args.mat_folder / '{}.mat'.format(foodweb)
        if not mat_file.is_file():
            message = 'Missing MAT file for "{}": {}'.format(foodweb, mat_file)
            if args.continue_on_error:
                print('[WARN] ' + message)
                failures.append((foodweb, message))
                continue
            raise FileNotFoundError(message)

        for experiment_id in range(1, args.num_experiments + 1):
            seed = args.base_seed + experiment_id - 1
            cmd = build_command(args, script_dir, foodweb, seed, experiment_id, test_ratio)
            log_file = log_dir / '{}_SEAL_exp{:03d}_seed{}.log'.format(
                safe_file_stem(foodweb), experiment_id, seed
            )

            print('[SEAL_directed] {} | experiment {}/{} | seed {} | log {}'.format(
                foodweb, experiment_id, args.num_experiments, seed, log_file
            ))
            if args.dry_run:
                print(' '.join('"{}"'.format(part) if ' ' in part else part for part in cmd))
                continue

            with open(log_file, 'w') as log:
                completed = subprocess.run(
                    cmd,
                    cwd=script_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            if completed.returncode != 0:
                message = 'SEAL_directed failed for "{}" experiment {}. See {}'.format(
                    foodweb, experiment_id, log_file
                )
                if args.continue_on_error:
                    print('[WARN] ' + message)
                    failures.append((foodweb, message))
                    continue
                raise RuntimeError(message)

    if failures:
        print('[SEAL_directed] Completed with {} failure(s).'.format(len(failures)))
        for foodweb, message in failures:
            print('[SEAL_directed] {}: {}'.format(foodweb, message))
    else:
        print('[SEAL_directed] Completed all requested food-web runs.')


if __name__ == '__main__':
    main()
