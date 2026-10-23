"""Common model interface for weak, directed, scalar-gated and ECC variants."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


VARIANTS = ("legacy_weak_node", "directed_node", "directed_scalar_gate", "directed_ecc")


def collate_directed(graphs, label_dim, device):
    graph_sizes = [graph.num_nodes for graph in graphs]
    labels = torch.tensor([graph.graph_label for graph in graphs], dtype=torch.long, device=device)
    node_labels = torch.cat(
        [torch.as_tensor(graph.node_labels, dtype=torch.long) for graph in graphs], dim=0
    )
    one_hot = torch.zeros((node_labels.numel(), int(label_dim)), dtype=torch.float32)
    one_hot.scatter_(1, node_labels.view(-1, 1), 1.0)
    attributes = torch.cat(
        [torch.as_tensor(graph.node_features, dtype=torch.float32) for graph in graphs], dim=0
    )
    node_input = torch.cat([one_hot, attributes], dim=1).to(device)
    edges, edge_attrs = [], []
    offset = 0
    for graph in graphs:
        if graph.edge_index.size:
            edges.append(torch.as_tensor(graph.edge_index, dtype=torch.long) + offset)
            edge_attrs.append(torch.as_tensor(graph.edge_attr, dtype=torch.float32))
        offset += graph.num_nodes
    edge_index = (
        torch.cat(edges, dim=0).t().contiguous().to(device)
        if edges else torch.empty((2, 0), dtype=torch.long, device=device)
    )
    edge_attr = (
        torch.cat(edge_attrs, dim=0).to(device)
        if edge_attrs else torch.empty((0, 17), dtype=torch.float32, device=device)
    )
    return node_input, edge_index, edge_attr, graph_sizes, labels


def mean_aggregate(messages, indices, nodes):
    if messages.numel() == 0:
        return messages.new_zeros((nodes, messages.shape[-1]))
    output = messages.new_zeros((nodes, messages.shape[-1]))
    output.index_add_(0, indices, messages)
    counts = messages.new_zeros((nodes, 1))
    counts.index_add_(0, indices, messages.new_ones((messages.shape[0], 1)))
    return output / counts.clamp_min(1.0)


class EdgeGate(nn.Module):
    def __init__(self, edge_dim=17):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(edge_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, edge_attr):
        return torch.sigmoid(self.network(edge_attr))


class DirectedLayer(nn.Module):
    def __init__(self, input_dim, output_dim, mode, edge_dim=17, edge_chunk_size=8192):
        super().__init__()
        if mode not in {"node", "scalar", "ecc"}:
            raise ValueError(mode)
        self.mode = mode
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.edge_chunk_size = int(edge_chunk_size)
        self.self_linear = nn.Linear(input_dim, output_dim, bias=True)
        if mode in {"node", "scalar"}:
            self.in_linear = nn.Linear(input_dim, output_dim, bias=False)
            self.out_linear = nn.Linear(input_dim, output_dim, bias=False)
        else:
            filter_size = input_dim * output_dim
            self.in_filter = nn.Sequential(
                nn.Linear(edge_dim, 32), nn.ReLU(), nn.Linear(32, filter_size)
            )
            self.out_filter = nn.Sequential(
                nn.Linear(edge_dim, 32), nn.ReLU(), nn.Linear(32, filter_size)
            )

    def _ecc_messages(self, features, attributes, network):
        chunks = []
        for start in range(0, features.shape[0], self.edge_chunk_size):
            stop = min(start + self.edge_chunk_size, features.shape[0])
            filters = network(attributes[start:stop]).view(
                -1, self.output_dim, self.input_dim
            )
            chunks.append(torch.bmm(filters, features[start:stop].unsqueeze(-1)).squeeze(-1))
        return torch.cat(chunks, dim=0) if chunks else features.new_zeros((0, self.output_dim))

    def forward(self, node_state, edge_index, edge_attr, scalar_gate=None):
        nodes = node_state.shape[0]
        if edge_index.numel() == 0:
            incoming = node_state.new_zeros((nodes, self.output_dim))
            outgoing = node_state.new_zeros((nodes, self.output_dim))
        else:
            source, target = edge_index[0], edge_index[1]
            if self.mode == "ecc":
                in_messages = self._ecc_messages(node_state[source], edge_attr, self.in_filter)
                out_messages = self._ecc_messages(node_state[target], edge_attr, self.out_filter)
            else:
                in_messages = self.in_linear(node_state[source])
                out_messages = self.out_linear(node_state[target])
                if self.mode == "scalar":
                    if scalar_gate is None:
                        raise ValueError("scalar gate is required")
                    in_messages = in_messages * scalar_gate
                    out_messages = out_messages * scalar_gate
            incoming = mean_aggregate(in_messages, target, nodes)
            outgoing = mean_aggregate(out_messages, source, nodes)
        return torch.tanh(self.self_linear(node_state) + incoming + outgoing)


class DirectedEncoder(nn.Module):
    def __init__(self, input_dim, variant, latent_dims=(32, 32, 32, 1), edge_dim=17,
                 edge_chunk_size=8192):
        super().__init__()
        mode = {
            "directed_node": "node",
            "directed_scalar_gate": "scalar",
            "directed_ecc": "ecc",
        }.get(variant)
        if mode is None:
            raise ValueError(variant)
        self.variant = variant
        self.scalar_gate = EdgeGate(edge_dim) if mode == "scalar" else None
        dimensions = [int(input_dim)] + [int(value) for value in latent_dims]
        self.layers = nn.ModuleList(
            [
                DirectedLayer(
                    dimensions[index], dimensions[index + 1], mode,
                    edge_dim=edge_dim, edge_chunk_size=edge_chunk_size,
                )
                for index in range(len(latent_dims))
            ]
        )
        self.total_latent_dim = int(sum(latent_dims))

    def forward(self, node_input, edge_index, edge_attr):
        state = node_input
        states = []
        gates = self.scalar_gate(edge_attr) if self.scalar_gate is not None else None
        for layer in self.layers:
            state = layer(state, edge_index, edge_attr, scalar_gate=gates)
            states.append(state)
        return torch.cat(states, dim=1), gates


class SortPoolingHead(nn.Module):
    def __init__(self, total_latent_dim, k, hidden=128, dropout=True):
        super().__init__()
        self.total_latent_dim = int(total_latent_dim)
        self.k = int(k)
        self.conv1 = nn.Conv1d(1, 16, self.total_latent_dim, self.total_latent_dim)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, 5, 1)
        dense_positions = int((self.k - 2) / 2 + 1) - 5 + 1
        if dense_positions < 1:
            raise ValueError("sort-pooling k={} is too small".format(self.k))
        self.linear1 = nn.Linear(dense_positions * 32, int(hidden))
        self.linear2 = nn.Linear(int(hidden), 2)
        self.dropout = bool(dropout)

    def forward(self, states, graph_sizes):
        pooled = states.new_zeros((len(graph_sizes), self.k, self.total_latent_dim))
        offset = 0
        for graph_index, size in enumerate(graph_sizes):
            graph_state = states[offset: offset + size]
            take = min(self.k, int(size))
            indices = torch.topk(graph_state[:, -1], take).indices
            pooled[graph_index, :take] = graph_state.index_select(0, indices)
            offset += int(size)
        values = pooled.reshape(-1, 1, self.k * self.total_latent_dim)
        values = F.relu(self.conv1(values))
        values = self.pool(values)
        values = F.relu(self.conv2(values))
        values = F.relu(self.linear1(values.reshape(len(graph_sizes), -1)))
        values = F.dropout(values, training=self.training) if self.dropout else values
        return F.log_softmax(self.linear2(values), dim=1)


class DirectedClassifier(nn.Module):
    def __init__(self, node_input_dim, label_dim, variant, sortpooling_k,
                 latent_dims=(32, 32, 32, 1), hidden=128, dropout=True,
                 edge_dim=17, edge_chunk_size=8192):
        super().__init__()
        self.label_dim = int(label_dim)
        self.encoder = DirectedEncoder(
            node_input_dim, variant, latent_dims, edge_dim, edge_chunk_size
        )
        self.head = SortPoolingHead(sum(latent_dims), sortpooling_k, hidden, dropout)
        self.last_gates = None

    def forward(self, graphs):
        device = next(self.parameters()).device
        node_input, edges, edge_attr, sizes, _ = collate_directed(graphs, self.label_dim, device)
        states, gates = self.encoder(node_input, edges, edge_attr)
        self.last_gates = gates
        return self.head(states, sizes)


def _to_legacy_graphs(graphs):
    import networkx as nx
    python_dir = Path(__file__).resolve().parents[1]
    backend = python_dir / "pytorch_DGCNN"
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    from util import GNNGraph

    result = []
    for graph in graphs:
        weak = nx.Graph()
        weak.add_nodes_from(range(graph.num_nodes))
        weak.add_edges_from((int(u), int(v)) for u, v in graph.edge_index)
        result.append(
            GNNGraph(
                weak,
                graph.graph_label,
                graph.node_labels.tolist(),
                np.asarray(graph.node_features, dtype=np.float32),
            )
        )
    return result


class LegacyWeakClassifier(nn.Module):
    """Existing DGCNN architecture fed the same weak-projection representation."""

    def __init__(self, node_input_dim, label_dim, sortpooling_k,
                 latent_dims=(32, 32, 32, 1), hidden=128, dropout=True):
        super().__init__()
        python_dir = Path(__file__).resolve().parents[1]
        backend = python_dir / "pytorch_DGCNN"
        if str(backend) not in sys.path:
            sys.path.insert(0, str(backend))
        from DGCNN_embedding import DGCNN
        from mlp_dropout import MLPClassifier

        self.label_dim = int(label_dim)
        self.gnn = DGCNN(
            output_dim=0,
            num_node_feats=int(node_input_dim),
            num_edge_feats=0,
            latent_dim=list(latent_dims),
            k=int(sortpooling_k),
        )
        self.mlp = MLPClassifier(
            input_size=self.gnn.dense_dim,
            hidden_size=int(hidden),
            num_class=2,
            with_dropout=bool(dropout),
        )

    def forward(self, graphs):
        legacy = _to_legacy_graphs(graphs)
        device = next(self.parameters()).device
        tags = torch.cat([torch.as_tensor(g.node_tags, dtype=torch.long) for g in legacy])
        one_hot = torch.zeros((tags.numel(), self.label_dim), dtype=torch.float32)
        one_hot.scatter_(1, tags.view(-1, 1), 1.0)
        attrs = torch.cat(
            [torch.as_tensor(g.node_features, dtype=torch.float32) for g in legacy], dim=0
        )
        node_input = torch.cat([one_hot, attrs], dim=1).to(device)
        embedding = self.gnn(legacy, node_input, None)
        return self.mlp(embedding)


def choose_sortpooling_k(graphs, fraction=0.6):
    sizes = sorted(graph.num_nodes for graph in graphs)
    index = max(0, int(math.ceil(float(fraction) * len(sizes))) - 1)
    return max(10, int(sizes[index]))


def build_classifier(variant, node_attribute_dim, label_dim, sortpooling_k,
                     edge_chunk_size=8192, latent_dims=(32, 32, 32, 1), hidden=128,
                     dropout=True):
    if variant not in VARIANTS:
        raise ValueError("unknown model variant {}".format(variant))
    node_input_dim = int(node_attribute_dim) + int(label_dim)
    if variant == "legacy_weak_node":
        return LegacyWeakClassifier(
            node_input_dim, label_dim, sortpooling_k, latent_dims, hidden, dropout
        )
    return DirectedClassifier(
        node_input_dim, label_dim, variant, sortpooling_k,
        latent_dims, hidden, dropout, edge_dim=17, edge_chunk_size=edge_chunk_size,
    )
