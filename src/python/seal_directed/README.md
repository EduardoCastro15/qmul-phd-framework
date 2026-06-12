# SEAL Directed

This folder contains the directed SEAL pipeline. It is intentionally separate
from the existing SEAL runner in `src/python`.

## What Is Directed

- Positive links are ordered pairs from `net[source, target]`.
- Test masking removes only `source -> target`, not `target -> source`.
- Negative links are ordered absent pairs where `source != target`.
- By default, negative sampling uses ecological role filtering:
  source nodes must be resources or consumer-resources, and target nodes must
  be consumers or consumer-resources.
- Source and target nodes receive distinct labels in the enclosing subgraph.

## Current Model Limitation

The bundled DGCNN backend builds undirected message-passing matrices. This
pipeline therefore uses directed candidates and directed masking, then trains
on the weak projection of each enclosing subgraph while preserving pair order
through source/target labels.

## Node Attributes

Node attributes are integrated only in this directed pipeline. The attributed
MAT files live in:

```text
src/python/seal_directed/data/foodwebs_mat_seal_attrs/
```

They contain the original `net`, `taxonomy`, `mass`, and `role`, plus `group`,
the node feature matrix read by `--use-attribute`.

Regenerate them with:

```bash
/Users/acw792/miniconda3/envs/Foodweb/bin/python \
  src/python/seal_directed/build_node_attribute_mats.py \
  --overwrite
```

## Example

```bash
conda activate Foodweb

python src/python/seal_directed/run_foodweb_seal_directed.py \
  --train-ratio 0.9 \
  --num-experiments 1 \
  --hop 1 \
  --only-foodweb "Tuesday Lake 1986_tax_mass" \
  --use-attribute \
  --no-parallel
```

If `python` resolves to the system interpreter, pass the environment explicitly:

```bash
python src/python/seal_directed/run_foodweb_seal_directed.py \
  --python-executable /Users/acw792/miniconda3/envs/Foodweb/bin/python \
  --train-ratio 0.9 \
  --num-experiments 1 \
  --hop 1 \
  --only-foodweb "Tuesday Lake 1986_tax_mass" \
  --use-attribute \
  --no-parallel
```

Results are written by default to:

```text
src/python/seal_directed/data/result/prediction_scores_logs/
```
