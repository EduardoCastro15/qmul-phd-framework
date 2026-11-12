"""Lossless ragged artifacts for directed subgraphs with node and edge features."""

from __future__ import annotations

import hashlib

import numpy as np

from subgraphs import DirectedSubgraph


NODE_ATTRIBUTE_DIM = 44
EDGE_ATTRIBUTE_DIM = 17


def _artifact_keys(artifact):
    return set(artifact.files if hasattr(artifact, "files") else artifact.keys())


def _empty(rows, columns, dtype):
    return np.empty((rows, columns), dtype=dtype)


def pack_graphs(graphs, prefix):
    """Pack graphs without reordering nodes or directed edges."""
    node_offsets = [0]
    edge_offsets = [0]
    node_ids = []
    node_labels = []
    node_features = []
    edge_index = []
    edge_attr = []
    graph_labels = []
    candidates = []

    for graph in graphs:
        ids = np.asarray(graph.node_ids, dtype=np.int64).reshape(-1)
        labels = np.asarray(graph.node_labels, dtype=np.int64).reshape(-1)
        features = np.asarray(graph.node_features, dtype=np.float32).reshape(
            -1, NODE_ATTRIBUTE_DIM
        )
        edges = np.asarray(graph.edge_index, dtype=np.int64).reshape(-1, 2)
        attributes = np.asarray(graph.edge_attr, dtype=np.float32).reshape(
            -1, EDGE_ATTRIBUTE_DIM
        )
        candidate = np.asarray(graph.candidate, dtype=np.int64).reshape(2)
        if ids.size != int(graph.num_nodes) or labels.size != ids.size:
            raise ValueError("Graph node metadata has inconsistent length")
        if features.shape[0] != ids.size:
            raise ValueError("Graph node features are not aligned with node order")
        if edges.shape[0] != attributes.shape[0]:
            raise ValueError("Graph edge attributes are not aligned with edge order")
        if ids.size < 2 or not np.array_equal(ids[:2], candidate):
            raise ValueError("Candidate endpoints must occupy canonical node positions 0 and 1")
        if edges.size and (edges.min() < 0 or edges.max() >= ids.size):
            raise ValueError("Graph contains an out-of-range local edge endpoint")
        if edges.size and np.any((edges[:, 0] == 0) & (edges[:, 1] == 1)):
            raise ValueError("Candidate edge leaked into a packed subgraph")
        if not np.all(np.isfinite(features)) or not np.all(np.isfinite(attributes)):
            raise ValueError("Packed graph inputs must be finite")

        node_ids.append(ids)
        node_labels.append(labels)
        node_features.append(features)
        edge_index.append(edges)
        edge_attr.append(attributes)
        graph_labels.append(int(graph.graph_label))
        candidates.append(candidate)
        node_offsets.append(node_offsets[-1] + ids.size)
        edge_offsets.append(edge_offsets[-1] + edges.shape[0])

    return {
        f"{prefix}_node_offsets": np.asarray(node_offsets, dtype=np.int64),
        f"{prefix}_edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        f"{prefix}_node_ids": (
            np.concatenate(node_ids) if node_ids else np.empty(0, dtype=np.int64)
        ),
        f"{prefix}_node_labels": (
            np.concatenate(node_labels) if node_labels else np.empty(0, dtype=np.int64)
        ),
        f"{prefix}_node_features": (
            np.concatenate(node_features, axis=0)
            if node_features else _empty(0, NODE_ATTRIBUTE_DIM, np.float32)
        ),
        f"{prefix}_edge_index": (
            np.concatenate(edge_index, axis=0)
            if edge_index else _empty(0, 2, np.int64)
        ),
        f"{prefix}_edge_attr": (
            np.concatenate(edge_attr, axis=0)
            if edge_attr else _empty(0, EDGE_ATTRIBUTE_DIM, np.float32)
        ),
        f"{prefix}_graph_labels": np.asarray(graph_labels, dtype=np.int8),
        f"{prefix}_candidate_pairs": (
            np.stack(candidates, axis=0) if candidates else _empty(0, 2, np.int64)
        ),
    }


def validate_packed_graphs(artifact, prefix, expected_pairs=None, expected_labels=None):
    required = {
        f"{prefix}_node_offsets",
        f"{prefix}_edge_offsets",
        f"{prefix}_node_ids",
        f"{prefix}_node_labels",
        f"{prefix}_node_features",
        f"{prefix}_edge_index",
        f"{prefix}_edge_attr",
        f"{prefix}_graph_labels",
        f"{prefix}_candidate_pairs",
    }
    missing = sorted(required.difference(_artifact_keys(artifact)))
    if missing:
        raise ValueError(f"Missing packed {prefix} graph arrays: {', '.join(missing)}")

    node_offsets = np.asarray(artifact[f"{prefix}_node_offsets"], dtype=np.int64)
    edge_offsets = np.asarray(artifact[f"{prefix}_edge_offsets"], dtype=np.int64)
    node_ids = np.asarray(artifact[f"{prefix}_node_ids"], dtype=np.int64).reshape(-1)
    labels = np.asarray(artifact[f"{prefix}_node_labels"], dtype=np.int64).reshape(-1)
    features = np.asarray(artifact[f"{prefix}_node_features"], dtype=np.float32)
    edges = np.asarray(artifact[f"{prefix}_edge_index"], dtype=np.int64).reshape(-1, 2)
    attributes = np.asarray(artifact[f"{prefix}_edge_attr"], dtype=np.float32)
    graph_labels = np.asarray(artifact[f"{prefix}_graph_labels"], dtype=np.int64).reshape(-1)
    candidates = np.asarray(artifact[f"{prefix}_candidate_pairs"], dtype=np.int64).reshape(-1, 2)
    graph_count = graph_labels.size

    if node_offsets.shape != (graph_count + 1,) or edge_offsets.shape != (graph_count + 1,):
        raise ValueError(f"Invalid {prefix} ragged offsets")
    if node_offsets[0] != 0 or edge_offsets[0] != 0:
        raise ValueError(f"{prefix} offsets must start at zero")
    if np.any(np.diff(node_offsets) < 2) or np.any(np.diff(edge_offsets) < 0):
        raise ValueError(f"Invalid {prefix} graph sizes")
    if node_offsets[-1] != node_ids.size or node_offsets[-1] != labels.size:
        raise ValueError(f"{prefix} node offsets do not cover node metadata")
    if features.shape != (node_ids.size, NODE_ATTRIBUTE_DIM):
        raise ValueError(f"Invalid {prefix} node feature shape")
    if edge_offsets[-1] != edges.shape[0]:
        raise ValueError(f"{prefix} edge offsets do not cover edge_index")
    if attributes.shape != (edges.shape[0], EDGE_ATTRIBUTE_DIM):
        raise ValueError(f"Invalid {prefix} edge attribute shape")
    if candidates.shape[0] != graph_count:
        raise ValueError(f"{prefix} candidate count does not match graph count")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(attributes)):
        raise ValueError(f"Non-finite packed {prefix} graph inputs")
    if expected_pairs is not None and not np.array_equal(candidates, np.asarray(expected_pairs)):
        raise ValueError(f"{prefix} candidates do not match the frozen split")
    if expected_labels is not None and not np.array_equal(graph_labels, np.asarray(expected_labels)):
        raise ValueError(f"{prefix} labels do not match expected labels")

    for index in range(graph_count):
        n0, n1 = map(int, node_offsets[index:index + 2])
        e0, e1 = map(int, edge_offsets[index:index + 2])
        local_edges = edges[e0:e1]
        if not np.array_equal(node_ids[n0:n0 + 2], candidates[index]):
            raise ValueError(f"{prefix} candidate/node ordering mismatch at graph {index}")
        size = n1 - n0
        if local_edges.size and (local_edges.min() < 0 or local_edges.max() >= size):
            raise ValueError(f"{prefix} local edge out of range at graph {index}")
        if local_edges.size and np.any(
            (local_edges[:, 0] == 0) & (local_edges[:, 1] == 1)
        ):
            raise ValueError(f"{prefix} candidate edge leaked at graph {index}")
    return graph_count


def unpack_graphs(artifact, prefix):
    """Reconstruct exact ``DirectedSubgraph`` objects without extraction."""
    validate_packed_graphs(artifact, prefix)
    node_offsets = np.asarray(artifact[f"{prefix}_node_offsets"], dtype=np.int64)
    edge_offsets = np.asarray(artifact[f"{prefix}_edge_offsets"], dtype=np.int64)
    node_ids = np.asarray(artifact[f"{prefix}_node_ids"], dtype=np.int64)
    node_labels = np.asarray(artifact[f"{prefix}_node_labels"], dtype=np.int64)
    node_features = np.asarray(artifact[f"{prefix}_node_features"], dtype=np.float32)
    edge_index = np.asarray(artifact[f"{prefix}_edge_index"], dtype=np.int64).reshape(-1, 2)
    edge_attr = np.asarray(artifact[f"{prefix}_edge_attr"], dtype=np.float32)
    graph_labels = np.asarray(artifact[f"{prefix}_graph_labels"], dtype=np.int64)
    candidates = np.asarray(artifact[f"{prefix}_candidate_pairs"], dtype=np.int64).reshape(-1, 2)
    result = []
    for index, graph_label in enumerate(graph_labels):
        n0, n1 = map(int, node_offsets[index:index + 2])
        e0, e1 = map(int, edge_offsets[index:index + 2])
        result.append(DirectedSubgraph(
            graph_label=int(graph_label),
            node_ids=node_ids[n0:n1].copy(),
            node_labels=node_labels[n0:n1].copy(),
            node_features=node_features[n0:n1].copy(),
            edge_index=edge_index[e0:e1].copy(),
            edge_attr=edge_attr[e0:e1].copy(),
            candidate=tuple(int(value) for value in candidates[index]),
        ))
    return result


def packed_graph_hash(arrays, prefixes=("train", "test")):
    """Hash packed inputs independently of the NPZ container metadata."""
    digest = hashlib.sha256()
    names = sorted(
        name for name in arrays
        if any(name.startswith(f"{prefix}_") for prefix in prefixes)
    )
    for name in names:
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()
