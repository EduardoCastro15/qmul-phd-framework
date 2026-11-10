# Jorge + aligned node features

This package contains the code used for our node-feature extension, before adding Logistic probability weighting.

Files:
- node_ecological_encoder.py: node attribute preprocessing and alignment to the canonical subgraph node order.
- run_node_ecological_features.py: actual experiment runner used in our project.
- wlnm.py: the local Python encoder/classifier implementation used by that runner, included to make its interface explicit.

Main change (runner lines 75–76):
    aligned = aligned_node_block(node_ids, features)
    x = np.column_stack([topology, aligned])

The original 90-dimensional structural representation is concatenated with attributes for the K=10 canonically ordered subgraph nodes. Attributes include log(1+biomass), log(mean individual weight), missingness indicators, one-hot feeding role, and a real-node/padding indicator. Preprocessing is fitted on unique nodes in training candidate subgraphs. Held-out labels are not used for preprocessing or fitting. The MLP input dimension increases; the remaining classifier configuration is retained.

Integration into another codebase:
1. Obtain the node IDs in exactly the same canonical order as the structural encoding.
2. Pad node IDs to K with -1.
3. Fit the preprocessor on training-subgraph nodes, then transform all required nodes.
4. Call aligned_node_block and concatenate it with the structural vector.
5. Set the classifier input dimension to the concatenated feature dimension and retrain.

IMPORTANT: This is a source-code handoff, not a standalone reproduction package. The runner also imports project-specific data preparation/evaluation modules and expects local data and saved baseline splits. Those dependencies are not included here. The node alignment helper calls the local Encoder.encode(pair, return_debug=True) interface; a different original implementation will need to expose its canonical node order. No claim is made that the recipient's local code has the same interface.

Experiment scope: within each year, nine spatial groups are aggregated into one Fold-level web. Candidate pairs are split within that web. This is not held-out-location transfer.
