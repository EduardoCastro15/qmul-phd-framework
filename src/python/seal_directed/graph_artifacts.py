"""Lossless ragged serialization of the fixed enclosing subgraphs."""

from __future__ import annotations

import networkx as nx
import numpy as np


def pack_graphs(graphs, prefix):
    node_offsets = [0]
    edge_offsets = [0]
    node_ids = []
    node_labels = []
    edge_pairs = []
    graph_labels = []
    candidate_pairs = []

    for graph in graphs:
        if not hasattr(graph, "node_ids") or not hasattr(graph, "candidate_pair"):
            raise ValueError("Graph is missing node_ids/candidate_pair provenance")
        ids = np.asarray(graph.node_ids, dtype=np.int64).reshape(-1)
        labels = np.asarray(graph.node_tags, dtype=np.int64).reshape(-1)
        if ids.size != int(graph.num_nodes) or labels.size != int(graph.num_nodes):
            raise ValueError("Graph node metadata has inconsistent length")
        edges = np.asarray(graph.edge_pairs, dtype=np.int64).reshape(-1, 2)
        node_ids.append(ids)
        node_labels.append(labels)
        edge_pairs.append(edges)
        graph_labels.append(int(graph.label))
        candidate_pairs.append(np.asarray(graph.candidate_pair, dtype=np.int64).reshape(2))
        node_offsets.append(node_offsets[-1] + ids.size)
        edge_offsets.append(edge_offsets[-1] + edges.shape[0])

    return {
        "{}_node_offsets".format(prefix): np.asarray(node_offsets, dtype=np.int64),
        "{}_node_ids".format(prefix): (
            np.concatenate(node_ids) if node_ids else np.empty(0, dtype=np.int64)
        ),
        "{}_node_labels".format(prefix): (
            np.concatenate(node_labels) if node_labels else np.empty(0, dtype=np.int64)
        ),
        "{}_edge_offsets".format(prefix): np.asarray(edge_offsets, dtype=np.int64),
        "{}_edge_pairs".format(prefix): (
            np.concatenate(edge_pairs, axis=0)
            if edge_pairs else np.empty((0, 2), dtype=np.int64)
        ),
        "{}_graph_labels".format(prefix): np.asarray(graph_labels, dtype=np.int8),
        "{}_candidate_pairs".format(prefix): (
            np.stack(candidate_pairs, axis=0)
            if candidate_pairs else np.empty((0, 2), dtype=np.int64)
        ),
    }


def validate_packed_graphs(artifact, prefix, expected_pairs=None, expected_labels=None):
    required = {
        "{}_node_offsets".format(prefix),
        "{}_node_ids".format(prefix),
        "{}_node_labels".format(prefix),
        "{}_edge_offsets".format(prefix),
        "{}_edge_pairs".format(prefix),
        "{}_graph_labels".format(prefix),
        "{}_candidate_pairs".format(prefix),
    }
    absent = sorted(required.difference(artifact.files))
    if absent:
        raise ValueError("Missing packed {} graph arrays: {}".format(prefix, ", ".join(absent)))

    node_offsets = np.asarray(artifact["{}_node_offsets".format(prefix)], dtype=np.int64)
    edge_offsets = np.asarray(artifact["{}_edge_offsets".format(prefix)], dtype=np.int64)
    graph_labels = np.asarray(artifact["{}_graph_labels".format(prefix)]).reshape(-1)
    candidates = np.asarray(artifact["{}_candidate_pairs".format(prefix)]).reshape(-1, 2)
    graph_count = graph_labels.size
    if node_offsets.shape != (graph_count + 1,) or edge_offsets.shape != (graph_count + 1,):
        raise ValueError("Invalid {} graph offsets".format(prefix))
    if node_offsets[0] != 0 or edge_offsets[0] != 0:
        raise ValueError("{} offsets must start at zero".format(prefix))
    if np.any(np.diff(node_offsets) < 2) or np.any(np.diff(edge_offsets) < 0):
        raise ValueError("Invalid ragged sizes for {} graphs".format(prefix))
    if node_offsets[-1] != np.asarray(artifact["{}_node_ids".format(prefix)]).size:
        raise ValueError("{} node offsets do not cover node_ids".format(prefix))
    if node_offsets[-1] != np.asarray(artifact["{}_node_labels".format(prefix)]).size:
        raise ValueError("{} node offsets do not cover node_labels".format(prefix))
    if edge_offsets[-1] != np.asarray(artifact["{}_edge_pairs".format(prefix)]).reshape(-1, 2).shape[0]:
        raise ValueError("{} edge offsets do not cover edge_pairs".format(prefix))
    if candidates.shape[0] != graph_count:
        raise ValueError("{} candidate count does not match graph count".format(prefix))
    if expected_pairs is not None and not np.array_equal(candidates, np.asarray(expected_pairs)):
        raise ValueError("{} candidates do not match split pairs".format(prefix))
    if expected_labels is not None and not np.array_equal(graph_labels, np.asarray(expected_labels)):
        raise ValueError("{} graph labels do not match expected labels".format(prefix))
    return graph_count


def unpack_graphs(artifact, prefix, node_attributes, graph_class):
    """Rebuild exact GNNGraph inputs without rerunning subgraph extraction."""
    validate_packed_graphs(artifact, prefix)
    node_offsets = np.asarray(artifact["{}_node_offsets".format(prefix)], dtype=np.int64)
    edge_offsets = np.asarray(artifact["{}_edge_offsets".format(prefix)], dtype=np.int64)
    all_ids = np.asarray(artifact["{}_node_ids".format(prefix)], dtype=np.int64)
    all_labels = np.asarray(artifact["{}_node_labels".format(prefix)], dtype=np.int64)
    all_edges = np.asarray(artifact["{}_edge_pairs".format(prefix)], dtype=np.int64).reshape(-1, 2)
    graph_labels = np.asarray(artifact["{}_graph_labels".format(prefix)], dtype=np.int64)
    candidates = np.asarray(artifact["{}_candidate_pairs".format(prefix)], dtype=np.int64).reshape(-1, 2)
    attributes = np.asarray(node_attributes, dtype=np.float32)
    graphs = []
    for index, graph_label in enumerate(graph_labels):
        n0, n1 = int(node_offsets[index]), int(node_offsets[index + 1])
        e0, e1 = int(edge_offsets[index]), int(edge_offsets[index + 1])
        node_ids = all_ids[n0:n1]
        local_edges = all_edges[e0:e1]
        graph_nx = nx.Graph()
        graph_nx.add_nodes_from(range(node_ids.size))
        graph_nx.add_edges_from((int(source), int(target)) for source, target in local_edges)
        graph = graph_class(
            graph_nx,
            int(graph_label),
            all_labels[n0:n1].tolist(),
            attributes[node_ids],
        )
        # NetworkX may emit the same undirected edge set in a different order
        # after reconstruction.  Preserve the exact frozen order because the
        # native float32 sparse reduction can otherwise differ at ~1e-7.
        graph.num_edges = int(local_edges.shape[0])
        graph.edge_pairs = local_edges.astype(np.int32, copy=True).reshape(-1)
        graph.node_ids = node_ids.copy()
        graph.candidate_pair = candidates[index].copy()
        graphs.append(graph)
    return graphs
