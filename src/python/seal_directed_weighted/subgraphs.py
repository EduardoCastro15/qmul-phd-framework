"""Leakage-safe directed enclosing subgraphs with aligned edge covariates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import scipy.sparse as ssp


@dataclass
class DirectedSubgraph:
    graph_label: int
    node_labels: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    candidate: tuple

    @property
    def num_nodes(self):
        return int(self.node_labels.shape[0])


def _weak_projection(matrix):
    weak = ssp.csr_matrix(matrix) + ssp.csr_matrix(matrix).transpose()
    weak.data = np.ones_like(weak.data)
    weak.setdiag(0)
    weak.eliminate_zeros()
    return weak


def _drnl(weak_subgraph):
    size = weak_subgraph.shape[0]
    without_0 = weak_subgraph[1:, 1:]
    keep_1 = [0] + list(range(2, size))
    without_1 = weak_subgraph[keep_1, :][:, keep_1]
    distance_0 = ssp.csgraph.shortest_path(without_0, directed=False, unweighted=True)[1:, 0]
    distance_1 = ssp.csgraph.shortest_path(without_1, directed=False, unweighted=True)[1:, 0]
    labels = np.zeros(distance_0.shape, dtype=np.int64)
    finite = np.isfinite(distance_0) & np.isfinite(distance_1)
    if finite.any():
        d0 = distance_0[finite].astype(np.int64)
        d1 = distance_1[finite].astype(np.int64)
        total = d0 + d1
        half, remainder = np.divmod(total, 2)
        labels[finite] = 1 + np.minimum(d0, d1) + half * (half + remainder - 1)
    labels = np.concatenate([np.asarray([1, 1], dtype=np.int64), labels])
    if labels.size > 2:
        mask = labels[2:] > 0
        labels[2:][mask] += 2
    labels[0], labels[1] = 1, 2
    return labels


def _neighbors(nodes, adjacency):
    result = set()
    for node in nodes:
        result.update(adjacency[:, node].nonzero()[0].tolist())
        result.update(adjacency[node, :].nonzero()[1].tolist())
    return result


def extract_one(candidate, graph_label, adjacency, node_features, context_edge_index,
                context_edge_attr, hop=1, max_nodes_per_hop=None,
                context_edge_lookup=None):
    source, target = map(int, candidate)
    if source == target:
        raise ValueError("candidate self-loop is not supported")
    adjacency = ssp.csr_matrix(adjacency)
    nodes = {source, target}
    visited = set(nodes)
    fringe = set(nodes)
    for _ in range(int(hop)):
        fringe = _neighbors(fringe, adjacency).difference(visited)
        if max_nodes_per_hop is not None and len(fringe) > int(max_nodes_per_hop):
            fringe = set(sorted(fringe)[: int(max_nodes_per_hop)])
        if not fringe:
            break
        visited.update(fringe)
        nodes.update(fringe)
    ordered = [source, target] + sorted(nodes.difference({source, target}))

    if context_edge_lookup is None:
        context_edge_lookup = {
            (int(edge[0]), int(edge[1])): index
            for index, edge in enumerate(np.asarray(context_edge_index, dtype=np.int64))
        }
    local_matrix = adjacency[ordered, :][:, ordered].tolil()
    local_matrix[0, 1] = 0
    local_matrix = local_matrix.tocsr()
    local_matrix.eliminate_zeros()
    local_row, local_col = local_matrix.nonzero()
    local_edges = np.column_stack([local_row, local_col]).astype(np.int64, copy=False)
    selected_attrs = []
    for local_source, local_target in local_edges:
        global_edge = (ordered[int(local_source)], ordered[int(local_target)])
        if global_edge not in context_edge_lookup:
            raise ValueError("context edge lacks covariates: {}".format(global_edge))
        selected_attrs.append(context_edge_attr[context_edge_lookup[global_edge]])
    local_attrs = np.asarray(selected_attrs, dtype=np.float32).reshape(-1, context_edge_attr.shape[1])
    if local_edges.size and np.any(
        (local_edges[:, 0] == 0) & (local_edges[:, 1] == 1)
    ):
        raise AssertionError("candidate edge leaked into enclosing subgraph")

    labels = _drnl(_weak_projection(local_matrix))
    features = np.asarray(node_features, dtype=np.float32)[ordered]
    return DirectedSubgraph(
        graph_label=int(graph_label),
        node_labels=labels,
        node_features=features,
        edge_index=local_edges,
        edge_attr=local_attrs,
        candidate=(source, target),
    )


def extract_dataset(adjacency, node_features, context_edge_index, context_edge_attr,
                    positive_pairs, negative_pairs, hop=1, max_nodes_per_hop=None,
                    num_workers=1):
    graphs = []
    context_edge_lookup = {
        (int(edge[0]), int(edge[1])): index
        for index, edge in enumerate(np.asarray(context_edge_index, dtype=np.int64))
    }
    if len(context_edge_lookup) != len(context_edge_index):
        raise ValueError("context_edge_index contains duplicates")
    for label, pairs in ((1, positive_pairs), (0, negative_pairs)):
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)

        def worker(pair):
            return extract_one(
                pair, label, adjacency, node_features,
                context_edge_index, context_edge_attr,
                hop=hop, max_nodes_per_hop=max_nodes_per_hop,
                context_edge_lookup=context_edge_lookup,
            )
        if int(num_workers) > 1 and len(pairs) > 1:
            with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
                graphs.extend(executor.map(worker, pairs))
        else:
            graphs.extend(worker(pair) for pair in pairs)
    return graphs


def reverse_graph(graph):
    return DirectedSubgraph(
        graph_label=graph.graph_label,
        node_labels=graph.node_labels.copy(),
        node_features=graph.node_features.copy(),
        edge_index=graph.edge_index[:, ::-1].copy(),
        edge_attr=graph.edge_attr.copy(),
        candidate=(graph.candidate[1], graph.candidate[0]),
    )

