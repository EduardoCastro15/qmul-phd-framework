"""Gradient attributions for the directed node and ECC SEAL variants.

The three input channels are deliberately attributed separately.  Node
attributes use role-matched empirical Expected Gradients, DRNL uses a zero
baseline Integrated Gradients path, and ECC edge covariates use complete
17-channel empirical edge vectors while keeping graph topology fixed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from models import collate_directed


CONFUSION_CODES = {"TN": 0, "FP": 1, "FN": 2, "TP": 3}
NODE_ROLE_CODES = {"focal_resource": 0, "focal_consumer": 1, "context": 2}
NODE_RNG_OFFSET = 10_000_019
EDGE_RNG_OFFSET = 20_000_033


def _candidate_pair(graph):
    value = getattr(graph, "candidate_pair", None)
    if value is None:
        value = getattr(graph, "candidate", None)
    if value is None:
        raise ValueError("graph is missing its candidate pair")
    return np.asarray(value, dtype=np.int64).reshape(2)


def _graph_label(graph):
    value = getattr(graph, "graph_label", getattr(graph, "label", None))
    if value is None:
        raise ValueError("graph is missing its binary label")
    return int(value)


def node_roles(node_count):
    if int(node_count) < 2:
        raise ValueError("an enclosing subgraph must contain both focal nodes")
    roles = np.full(int(node_count), NODE_ROLE_CODES["context"], dtype=np.int8)
    roles[0] = NODE_ROLE_CODES["focal_resource"]
    roles[1] = NODE_ROLE_CODES["focal_consumer"]
    return roles


def one_hot_drnl(graph, width, device):
    labels = torch.as_tensor(graph.node_labels, dtype=torch.long, device=device)
    if labels.numel() != int(graph.num_nodes):
        raise ValueError("DRNL label count does not match graph nodes")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= int(width)):
        raise ValueError("DRNL label is outside label_dim")
    result = torch.zeros((labels.numel(), int(width)), dtype=torch.float32, device=device)
    result.scatter_(1, labels.view(-1, 1), 1.0)
    return result


def _local_edge_index(graph, device):
    edges = np.asarray(graph.edge_index, dtype=np.int64).reshape(-1, 2)
    return torch.as_tensor(edges.T.copy(), dtype=torch.long, device=device)


def _repeat_edge_index(edge_index, node_count, batch):
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    offsets = torch.arange(batch, dtype=edge_index.dtype, device=edge_index.device)
    offsets = (offsets * int(node_count)).view(batch, 1, 1)
    repeated = edge_index.unsqueeze(0) + offsets
    return repeated.permute(1, 0, 2).reshape(2, -1)


class NodeAttributeLogOdds(nn.Module):
    def __init__(self, classifier, graph, fixed_drnl, fixed_edge_attributes):
        super().__init__()
        self.classifier = classifier
        self.node_count = int(graph.num_nodes)
        self.register_buffer("fixed_drnl", fixed_drnl.detach().clone())
        self.register_buffer("edge_index", _local_edge_index(graph, fixed_drnl.device))
        self.register_buffer(
            "fixed_edge_attributes", fixed_edge_attributes.detach().clone()
        )

    def forward(self, node_attributes):
        if node_attributes.ndim == 2:
            node_attributes = node_attributes.unsqueeze(0)
        if node_attributes.ndim != 3 or node_attributes.shape[1] != self.node_count:
            raise ValueError("Expected node attributes with shape [batch, nodes, features]")
        batch = int(node_attributes.shape[0])
        drnl = self.fixed_drnl.unsqueeze(0).expand(batch, -1, -1).reshape(
            batch * self.node_count, -1
        )
        attributes = node_attributes.reshape(batch * self.node_count, -1)
        edges = _repeat_edge_index(self.edge_index, self.node_count, batch)
        edge_attributes = self.fixed_edge_attributes.unsqueeze(0).expand(
            batch, -1, -1
        ).reshape(-1, self.fixed_edge_attributes.shape[1])
        return self.classifier.forward_log_odds(
            drnl, attributes, edges, edge_attributes, [self.node_count] * batch
        )


class DrnlLogOdds(nn.Module):
    def __init__(self, classifier, graph, fixed_node_attributes, fixed_edge_attributes):
        super().__init__()
        self.classifier = classifier
        self.node_count = int(graph.num_nodes)
        self.register_buffer(
            "fixed_node_attributes", fixed_node_attributes.detach().clone()
        )
        self.register_buffer("edge_index", _local_edge_index(graph, fixed_node_attributes.device))
        self.register_buffer(
            "fixed_edge_attributes", fixed_edge_attributes.detach().clone()
        )

    def forward(self, drnl_onehot):
        if drnl_onehot.ndim == 2:
            drnl_onehot = drnl_onehot.unsqueeze(0)
        if drnl_onehot.ndim != 3 or drnl_onehot.shape[1] != self.node_count:
            raise ValueError("Expected DRNL with shape [batch, nodes, labels]")
        batch = int(drnl_onehot.shape[0])
        drnl = drnl_onehot.reshape(batch * self.node_count, -1)
        attributes = self.fixed_node_attributes.unsqueeze(0).expand(
            batch, -1, -1
        ).reshape(batch * self.node_count, -1)
        edges = _repeat_edge_index(self.edge_index, self.node_count, batch)
        edge_attributes = self.fixed_edge_attributes.unsqueeze(0).expand(
            batch, -1, -1
        ).reshape(-1, self.fixed_edge_attributes.shape[1])
        return self.classifier.forward_log_odds(
            drnl, attributes, edges, edge_attributes, [self.node_count] * batch
        )


class EdgeAttributeLogOdds(nn.Module):
    def __init__(self, classifier, graph, fixed_drnl, fixed_node_attributes):
        super().__init__()
        self.classifier = classifier
        self.node_count = int(graph.num_nodes)
        self.edge_count = int(np.asarray(graph.edge_index).reshape(-1, 2).shape[0])
        self.register_buffer("fixed_drnl", fixed_drnl.detach().clone())
        self.register_buffer(
            "fixed_node_attributes", fixed_node_attributes.detach().clone()
        )
        self.register_buffer("edge_index", _local_edge_index(graph, fixed_drnl.device))

    def forward(self, edge_attributes):
        if edge_attributes.ndim == 2:
            edge_attributes = edge_attributes.unsqueeze(0)
        if edge_attributes.ndim != 3 or edge_attributes.shape[1] != self.edge_count:
            raise ValueError("Expected edge attributes with shape [batch, edges, features]")
        batch = int(edge_attributes.shape[0])
        drnl = self.fixed_drnl.unsqueeze(0).expand(batch, -1, -1).reshape(
            batch * self.node_count, -1
        )
        attributes = self.fixed_node_attributes.unsqueeze(0).expand(
            batch, -1, -1
        ).reshape(batch * self.node_count, -1)
        edges = _repeat_edge_index(self.edge_index, self.node_count, batch)
        return self.classifier.forward_log_odds(
            drnl,
            attributes,
            edges,
            edge_attributes.reshape(-1, edge_attributes.shape[-1]),
            [self.node_count] * batch,
        )


def build_node_backgrounds(train_graphs, query_nodes, samples, rng):
    """Sample training-only node baselines matched by focal/context position."""
    if int(samples) < 1 or not train_graphs:
        raise ValueError("Expected Gradients requires training backgrounds")
    required_context = int(query_nodes) - 2
    eligible = [
        index for index, graph in enumerate(train_graphs)
        if required_context == 0 or int(graph.num_nodes) > 2
    ]
    if not eligible:
        raise ValueError("no training subgraph contains context nodes")
    reference_indices = rng.choice(eligible, size=int(samples), replace=True)
    width = int(np.asarray(train_graphs[eligible[0]].node_features).shape[1])
    baselines = np.empty((int(samples), int(query_nodes), width), dtype=np.float32)
    for sample_index, graph_index in enumerate(reference_indices):
        reference = np.asarray(
            train_graphs[int(graph_index)].node_features, dtype=np.float32
        )
        if reference.shape[1] != width:
            raise ValueError("training node feature widths differ")
        baselines[sample_index, 0] = reference[0]
        baselines[sample_index, 1] = reference[1]
        if required_context:
            context = reference[2:]
            replace = context.shape[0] < required_context
            chosen = rng.choice(
                context.shape[0], size=required_context, replace=replace
            )
            baselines[sample_index, 2:] = context[chosen]
    return baselines, np.asarray(reference_indices, dtype=np.int64)


@dataclass(frozen=True)
class EdgeBackgroundPool:
    vectors: np.ndarray
    role_pairs: np.ndarray
    sha256: str


def build_edge_background_pool(train_graphs):
    """Build a deterministic training-only pool of complete edge vectors."""
    vectors, roles = [], []
    width = None
    for graph in train_graphs:
        edge_index = np.asarray(graph.edge_index, dtype=np.int64).reshape(-1, 2)
        edge_attr = np.asarray(graph.edge_attr, dtype=np.float32)
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[0]:
            raise ValueError("training graph edge attributes are misaligned")
        width = edge_attr.shape[1] if width is None else width
        if edge_attr.shape[1] != width:
            raise ValueError("training edge feature widths differ")
        local_roles = node_roles(graph.num_nodes)
        if edge_index.size:
            vectors.append(edge_attr)
            roles.append(local_roles[edge_index])
    if vectors:
        vector_array = np.concatenate(vectors).astype(np.float32, copy=False)
        role_array = np.concatenate(roles).astype(np.int8, copy=False)
    else:
        vector_array = np.empty((0, int(width or 17)), dtype=np.float32)
        role_array = np.empty((0, 2), dtype=np.int8)
    digest = hashlib.sha256()
    digest.update(np.asarray(vector_array.shape, dtype=np.int64).tobytes())
    digest.update(vector_array.tobytes(order="C"))
    digest.update(role_array.tobytes(order="C"))
    return EdgeBackgroundPool(vector_array, role_array, digest.hexdigest())


def sample_edge_backgrounds(pool, query_edge_index, query_node_count, samples, rng):
    """Sample whole edge vectors using exact-role, partial-role, then global fallback."""
    edges = np.asarray(query_edge_index, dtype=np.int64).reshape(-1, 2)
    if int(samples) < 1:
        raise ValueError("edge Expected Gradients requires at least one sample")
    if not len(edges):
        return (
            np.empty((int(samples), 0, pool.vectors.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.int8),
        )
    if pool.vectors.shape[0] == 0:
        raise ValueError("edge attribution requires training context edges")
    query_roles = node_roles(query_node_count)[edges]
    baselines = np.empty(
        (int(samples), len(edges), pool.vectors.shape[1]), dtype=np.float32
    )
    fallback_tiers = np.empty(len(edges), dtype=np.int8)
    global_indices = np.arange(pool.vectors.shape[0], dtype=np.int64)
    for edge_position, role_pair in enumerate(query_roles):
        exact = np.flatnonzero(np.all(pool.role_pairs == role_pair, axis=1))
        if exact.size:
            eligible, tier = exact, 0
        else:
            partial = np.flatnonzero(
                (pool.role_pairs[:, 0] == role_pair[0])
                | (pool.role_pairs[:, 1] == role_pair[1])
            )
            if partial.size:
                eligible, tier = partial, 1
            else:
                eligible, tier = global_indices, 2
        selected = rng.choice(eligible, size=int(samples), replace=True)
        baselines[:, edge_position] = pool.vectors[selected]
        fallback_tiers[edge_position] = tier
    return baselines, fallback_tiers


def expected_gradients(wrapper, inputs, baselines, rng, internal_batch_size=32):
    if inputs.ndim != 2 or baselines.ndim != 3 or baselines.shape[1:] != inputs.shape:
        raise ValueError("Expected Gradients input/baseline shape mismatch")
    samples = int(baselines.shape[0])
    if samples < 1:
        raise ValueError("Expected Gradients needs at least one baseline")
    batch_size = max(1, int(internal_batch_size))
    alpha = torch.as_tensor(
        rng.random((samples, 1, 1)), dtype=inputs.dtype, device=inputs.device
    )
    baseline_tensor = torch.as_tensor(
        baselines, dtype=inputs.dtype, device=inputs.device
    )
    total = torch.zeros_like(inputs)
    baseline_outputs = []
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        current = baseline_tensor[start:stop]
        interpolated = (
            current + alpha[start:stop] * (inputs.unsqueeze(0) - current)
        ).detach().requires_grad_(True)
        outputs = wrapper(interpolated)
        gradients = torch.autograd.grad(outputs.sum(), interpolated)[0]
        if not torch.all(torch.isfinite(gradients)):
            raise ValueError("non-finite Expected Gradients derivative")
        total += torch.sum((inputs.unsqueeze(0) - current) * gradients, dim=0)
        with torch.no_grad():
            baseline_outputs.append(wrapper(current).detach())
    attributions = total / float(samples)
    with torch.no_grad():
        input_output = wrapper(inputs.unsqueeze(0))[0]
        mean_baseline_output = torch.cat(baseline_outputs).mean()
    return attributions, _diagnostics(
        input_output, mean_baseline_output, attributions.sum()
    )


def integrated_gradients_drnl(wrapper, inputs, steps, internal_batch_size=32):
    """Gauss-Legendre Integrated Gradients from an all-zero DRNL baseline."""
    if inputs.ndim != 2 or int(steps) < 1:
        raise ValueError("DRNL Integrated Gradients expects [nodes, labels] and steps >= 1")
    nodes, weights = np.polynomial.legendre.leggauss(int(steps))
    alpha = torch.as_tensor(
        (nodes + 1.0) / 2.0, dtype=inputs.dtype, device=inputs.device
    ).view(-1, 1, 1)
    weights = torch.as_tensor(
        weights / 2.0, dtype=inputs.dtype, device=inputs.device
    ).view(-1, 1, 1)
    baseline = torch.zeros_like(inputs)
    total_gradient = torch.zeros_like(inputs)
    batch_size = max(1, int(internal_batch_size))
    for start in range(0, int(steps), batch_size):
        stop = min(start + batch_size, int(steps))
        interpolated = (
            baseline.unsqueeze(0)
            + alpha[start:stop] * (inputs - baseline).unsqueeze(0)
        ).detach().requires_grad_(True)
        outputs = wrapper(interpolated)
        gradients = torch.autograd.grad(outputs.sum(), interpolated)[0]
        if not torch.all(torch.isfinite(gradients)):
            raise ValueError("non-finite DRNL Integrated Gradients derivative")
        total_gradient += torch.sum(weights[start:stop] * gradients, dim=0)
    attributions = (inputs - baseline) * total_gradient
    with torch.no_grad():
        input_output = wrapper(inputs.unsqueeze(0))[0]
        baseline_output = wrapper(baseline.unsqueeze(0))[0]
    return attributions, _diagnostics(
        input_output, baseline_output, attributions.sum()
    )


def _diagnostics(input_output, baseline_output, attribution_sum):
    difference = input_output - baseline_output
    absolute = torch.abs(attribution_sum - difference)
    relative = absolute / torch.clamp(torch.abs(difference), min=1e-12)
    return {
        "input_output": float(input_output.detach().cpu()),
        "baseline_output": float(baseline_output.detach().cpu()),
        "output_difference": float(difference.detach().cpu()),
        "attribution_sum": float(attribution_sum.detach().cpu()),
        "completeness_absolute": float(absolute.detach().cpu()),
        "completeness_relative": float(relative.detach().cpu()),
    }


def validate_explicit_forward_parity(classifier, graphs, tolerance=1e-7):
    classifier.eval()
    device = next(classifier.parameters()).device
    with torch.no_grad():
        ordinary = classifier(graphs)
        packed, edges, edge_attr, sizes, _ = collate_directed(
            graphs, classifier.label_dim, device
        )
        explicit = classifier.forward_prepared(
            packed[:, : classifier.label_dim],
            packed[:, classifier.label_dim :],
            edges,
            edge_attr,
            sizes,
        )
    maximum = float(torch.max(torch.abs(ordinary - explicit)).cpu())
    if maximum > float(tolerance):
        raise ValueError("explicit forward parity failed: {:.12g}".format(maximum))
    return maximum


def validate_directed_node_edge_invariance(classifier, graphs, seed=0):
    if classifier.variant != "directed_node":
        raise ValueError("edge invariance is specific to directed_node")
    classifier.eval()
    device = next(classifier.parameters()).device
    packed, edges, edge_attr, sizes, _ = collate_directed(
        graphs, classifier.label_dim, device
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(edge_attr.shape[0], generator=generator).to(device)
    permuted = edge_attr.index_select(0, order) if edge_attr.shape[0] else edge_attr
    with torch.no_grad():
        original = classifier.forward_packed(packed, edges, edge_attr, sizes)
        changed = classifier.forward_packed(packed, edges, permuted, sizes)
    maximum = float(torch.max(torch.abs(original - changed)).cpu())
    if maximum != 0.0:
        raise ValueError("directed_node changed with edge attributes: {}".format(maximum))
    return maximum


def _prediction(classifier, graph, drnl, node_attributes, edge_attributes):
    edges = _local_edge_index(graph, drnl.device)
    with torch.no_grad():
        log_probabilities = classifier.forward_prepared(
            drnl,
            node_attributes,
            edges,
            edge_attributes,
            [int(graph.num_nodes)],
        )[0]
    return (
        float(torch.exp(log_probabilities[1]).cpu()),
        float((log_probabilities[1] - log_probabilities[0]).cpu()),
    )


def _confusion_code(label, score, threshold):
    prediction = int(float(score) > float(threshold))
    if int(label) == 1:
        return CONFUSION_CODES["TP" if prediction else "FN"]
    return CONFUSION_CODES["FP" if prediction else "TN"]


def _diag_array(diagnostics, key):
    return np.asarray([item[key] for item in diagnostics], dtype=np.float64)


def _absolute_proportion(values):
    denominator = float(np.sum(np.abs(values)))
    return np.abs(values) / denominator if denominator > 0 else np.zeros_like(values)


def _family_order(labels, expected_width, channel):
    if labels is None or len(labels) != int(expected_width):
        raise ValueError(
            "{} faithfulness requires {} feature-family labels".format(
                channel, expected_width
            )
        )
    labels = [str(value) for value in labels]
    if any(not value for value in labels):
        raise ValueError("{} feature-family labels cannot be empty".format(channel))
    return list(dict.fromkeys(labels)), labels


def _family_replacement_changes(
    wrapper,
    inputs,
    baselines,
    attribution_values,
    family_labels,
    family_order,
    samples,
):
    """Return top/bottom family indices and absolute log-odds changes."""
    importance = [
        float(np.sum(np.abs(attribution_values[:, np.asarray(
            [label == family for label in family_labels], dtype=bool
        )])))
        for family in family_order
    ]
    top_index = int(np.argmax(importance))
    bottom_index = int(np.argmin(importance))
    sample_count = min(int(samples), int(baselines.shape[0]))
    if sample_count < 1:
        raise ValueError("faithfulness replacement requires at least one baseline")
    baseline_tensor = torch.as_tensor(
        baselines[:sample_count], dtype=inputs.dtype, device=inputs.device
    )
    original = inputs.unsqueeze(0).expand(sample_count, -1, -1)
    with torch.no_grad():
        reference = wrapper(original)
    changes = []
    for family_index in (top_index, bottom_index):
        columns = torch.as_tensor(
            [label == family_order[family_index] for label in family_labels],
            dtype=torch.bool,
            device=inputs.device,
        )
        perturbed = original.clone()
        perturbed[:, :, columns] = baseline_tensor[:, :, columns]
        with torch.no_grad():
            altered = wrapper(perturbed)
        changes.append(float(torch.median(torch.abs(reference - altered)).cpu()))
    return top_index, bottom_index, changes[0], changes[1]


def attribute_test_graphs(
    classifier,
    variant,
    train_graphs,
    test_graphs,
    expected_scores,
    label_dim,
    attribution_seed,
    node_samples,
    drnl_steps,
    edge_samples=None,
    threshold=0.5,
    internal_batch_size=32,
    score_tolerance=1e-7,
    node_feature_families=None,
    edge_feature_families=None,
    faithfulness_background_samples=0,
):
    """Attribute every test candidate once and return ragged-array components."""
    if variant not in {"directed_node", "directed_ecc"}:
        raise ValueError("attribution supports directed_node and directed_ecc only")
    if classifier.variant != variant:
        raise ValueError("classifier variant disagrees with attribution request")
    faithfulness_background_samples = int(faithfulness_background_samples)
    if faithfulness_background_samples < 0:
        raise ValueError("faithfulness_background_samples cannot be negative")
    if faithfulness_background_samples:
        node_family_order, node_family_labels = _family_order(
            node_feature_families, classifier.node_attribute_dim, "node"
        )
        if variant == "directed_ecc":
            edge_family_order, edge_family_labels = _family_order(
                edge_feature_families, classifier.edge_dim, "edge"
            )
        else:
            edge_family_order, edge_family_labels = [], []
    else:
        node_family_order, node_family_labels = [], []
        edge_family_order, edge_family_labels = [], []
    classifier.eval()
    device = next(classifier.parameters()).device
    scores_expected = None
    if expected_scores is not None:
        scores_expected = np.asarray(expected_scores, dtype=np.float64).reshape(-1)
        if scores_expected.size != len(test_graphs):
            raise ValueError("stored score count does not match test graphs")
    if variant == "directed_ecc":
        edge_samples = int(node_samples if edge_samples is None else edge_samples)
        edge_pool = build_edge_background_pool(train_graphs)
    else:
        edge_samples = 0
        edge_pool = EdgeBackgroundPool(
            np.empty((0, 17), dtype=np.float32),
            np.empty((0, 2), dtype=np.int8),
            hashlib.sha256(b"directed_node_no_edge_pool").hexdigest(),
        )

    node_offsets, edge_offsets = [0], [0]
    candidate_pairs, labels, scores, log_odds, confusion = [], [], [], [], []
    all_node_ids, all_node_features, all_node_roles, all_drnl_labels = [], [], [], []
    node_values, node_normalized, node_background_indices = [], [], []
    drnl_values, drnl_normalized, drnl_active = [], [], []
    edge_indices, edge_features = [], []
    edge_values, edge_normalized, edge_roles, edge_fallbacks, edge_seeds = [], [], [], [], []
    node_diagnostics, drnl_diagnostics, edge_diagnostics = [], [], []
    faithfulness_node_top, faithfulness_node_bottom = [], []
    faithfulness_node_top_change, faithfulness_node_bottom_change = [], []
    faithfulness_edge_top, faithfulness_edge_bottom = [], []
    faithfulness_edge_top_change, faithfulness_edge_bottom_change = [], []

    for candidate_index, graph in enumerate(test_graphs):
        drnl = one_hot_drnl(graph, label_dim, device)
        node_attributes = torch.as_tensor(
            np.asarray(graph.node_features, dtype=np.float32), device=device
        )
        edge_attributes = torch.as_tensor(
            np.asarray(graph.edge_attr, dtype=np.float32), device=device
        )
        score, current_log_odds = _prediction(
            classifier, graph, drnl, node_attributes, edge_attributes
        )
        if scores_expected is not None and not math.isclose(
            score,
            float(scores_expected[candidate_index]),
            rel_tol=0.0,
            abs_tol=float(score_tolerance),
        ):
            raise ValueError(
                "checkpoint score mismatch for candidate {}: {:.12g} vs {:.12g}".format(
                    candidate_index, scores_expected[candidate_index], score
                )
            )

        node_seed = int(attribution_seed) + NODE_RNG_OFFSET + 1_000_003 * candidate_index
        node_rng = np.random.default_rng(node_seed)
        node_baselines, selected_backgrounds = build_node_backgrounds(
            train_graphs, graph.num_nodes, node_samples, node_rng
        )
        node_attr, node_diag = expected_gradients(
            NodeAttributeLogOdds(classifier, graph, drnl, edge_attributes),
            node_attributes,
            node_baselines,
            node_rng,
            internal_batch_size,
        )
        drnl_attr, drnl_diag = integrated_gradients_drnl(
            DrnlLogOdds(classifier, graph, node_attributes, edge_attributes),
            drnl,
            drnl_steps,
            internal_batch_size,
        )

        node_np = node_attr.detach().cpu().numpy().astype(np.float32)
        drnl_np = drnl_attr.detach().cpu().numpy().astype(np.float32)
        active_labels = np.asarray(graph.node_labels, dtype=np.int64)
        if faithfulness_background_samples:
            node_faithfulness = _family_replacement_changes(
                NodeAttributeLogOdds(classifier, graph, drnl, edge_attributes),
                node_attributes,
                node_baselines,
                node_np,
                node_family_labels,
                node_family_order,
                faithfulness_background_samples,
            )
            faithfulness_node_top.append(node_faithfulness[0])
            faithfulness_node_bottom.append(node_faithfulness[1])
            faithfulness_node_top_change.append(node_faithfulness[2])
            faithfulness_node_bottom_change.append(node_faithfulness[3])

        local_edges = np.asarray(graph.edge_index, dtype=np.int64).reshape(-1, 2)
        local_roles = node_roles(graph.num_nodes)
        if variant == "directed_ecc":
            edge_seed = int(attribution_seed) + EDGE_RNG_OFFSET + 1_000_003 * candidate_index
            edge_rng = np.random.default_rng(edge_seed)
            if len(local_edges):
                edge_baselines, fallback_tiers = sample_edge_backgrounds(
                    edge_pool, local_edges, graph.num_nodes, edge_samples, edge_rng
                )
                edge_attr, edge_diag = expected_gradients(
                    EdgeAttributeLogOdds(classifier, graph, drnl, node_attributes),
                    edge_attributes,
                    edge_baselines,
                    edge_rng,
                    internal_batch_size,
                )
                edge_np = edge_attr.detach().cpu().numpy().astype(np.float32)
                if faithfulness_background_samples:
                    edge_faithfulness = _family_replacement_changes(
                        EdgeAttributeLogOdds(classifier, graph, drnl, node_attributes),
                        edge_attributes,
                        edge_baselines,
                        edge_np,
                        edge_family_labels,
                        edge_family_order,
                        faithfulness_background_samples,
                    )
                    faithfulness_edge_top.append(edge_faithfulness[0])
                    faithfulness_edge_bottom.append(edge_faithfulness[1])
                    faithfulness_edge_top_change.append(edge_faithfulness[2])
                    faithfulness_edge_bottom_change.append(edge_faithfulness[3])
            else:
                fallback_tiers = np.empty(0, dtype=np.int8)
                edge_np = np.empty((0, classifier.edge_dim), dtype=np.float32)
                edge_diag = {
                    "input_output": current_log_odds,
                    "baseline_output": current_log_odds,
                    "output_difference": 0.0,
                    "attribution_sum": 0.0,
                    "completeness_absolute": 0.0,
                    "completeness_relative": 0.0,
                }
                if faithfulness_background_samples:
                    faithfulness_edge_top.append(-1)
                    faithfulness_edge_bottom.append(-1)
                    faithfulness_edge_top_change.append(0.0)
                    faithfulness_edge_bottom_change.append(0.0)
            edge_diagnostics.append(edge_diag)
        else:
            edge_seed = 0
            fallback_tiers = np.empty(0, dtype=np.int8)
            edge_np = np.empty((0, classifier.edge_dim), dtype=np.float32)

        stored_score = (
            float(scores_expected[candidate_index]) if scores_expected is not None else score
        )
        candidate_pairs.append(_candidate_pair(graph))
        graph_label = _graph_label(graph)
        labels.append(graph_label)
        scores.append(stored_score)
        log_odds.append(current_log_odds)
        confusion.append(_confusion_code(graph_label, stored_score, threshold))
        graph_node_ids = getattr(graph, "node_ids", None)
        if graph_node_ids is None:
            graph_node_ids = np.arange(int(graph.num_nodes), dtype=np.int64)
        graph_node_ids = np.asarray(graph_node_ids, dtype=np.int64).reshape(-1)
        if graph_node_ids.size != int(graph.num_nodes):
            raise ValueError("graph node_ids do not align with node features")
        all_node_ids.append(graph_node_ids)
        all_node_features.append(np.asarray(graph.node_features, dtype=np.float32))
        all_node_roles.append(local_roles)
        all_drnl_labels.append(active_labels)
        node_values.append(node_np)
        node_normalized.append(_absolute_proportion(node_np).astype(np.float32))
        node_background_indices.append(selected_backgrounds)
        drnl_values.append(drnl_np)
        drnl_normalized.append(_absolute_proportion(drnl_np).astype(np.float32))
        drnl_active.append(drnl_np[np.arange(len(active_labels)), active_labels])
        if variant == "directed_ecc":
            edge_indices.append(local_edges)
            edge_features.append(np.asarray(graph.edge_attr, dtype=np.float32))
            edge_values.append(edge_np)
            edge_normalized.append(_absolute_proportion(edge_np).astype(np.float32))
            edge_roles.append(
                local_roles[local_edges]
                if len(local_edges) else np.empty((0, 2), dtype=np.int8)
            )
            edge_fallbacks.append(fallback_tiers)
        edge_seeds.append(edge_seed)
        node_diagnostics.append(node_diag)
        drnl_diagnostics.append(drnl_diag)
        node_offsets.append(node_offsets[-1] + int(graph.num_nodes))
        edge_offsets.append(
            edge_offsets[-1] + (len(local_edges) if variant == "directed_ecc" else 0)
        )

    node_width = int(classifier.node_attribute_dim)
    edge_width = int(classifier.edge_dim)
    result = {
        "candidate_pairs": np.asarray(candidate_pairs, dtype=np.int64).reshape(-1, 2),
        "candidate_labels": np.asarray(labels, dtype=np.int8),
        "candidate_scores": np.asarray(scores, dtype=np.float64),
        "candidate_log_odds": np.asarray(log_odds, dtype=np.float64),
        "candidate_confusion": np.asarray(confusion, dtype=np.int8),
        "node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "node_ids": np.concatenate(all_node_ids).astype(np.int64) if all_node_ids else np.empty(0, np.int64),
        "node_feature_values": np.concatenate(all_node_features).reshape(-1, node_width).astype(np.float32) if all_node_features else np.empty((0, node_width), np.float32),
        "node_roles": np.concatenate(all_node_roles).astype(np.int8) if all_node_roles else np.empty(0, np.int8),
        "node_expected_gradients": np.concatenate(node_values).reshape(-1, node_width),
        "node_expected_gradients_abs_proportion": np.concatenate(node_normalized).reshape(-1, node_width),
        "node_background_train_indices": np.stack(node_background_indices).astype(np.int64) if node_background_indices else np.empty((0, int(node_samples)), np.int64),
        "drnl_labels": np.concatenate(all_drnl_labels).astype(np.int64) if all_drnl_labels else np.empty(0, np.int64),
        "drnl_integrated_gradients": np.concatenate(drnl_values).reshape(-1, int(label_dim)),
        "drnl_integrated_gradients_abs_proportion": np.concatenate(
            drnl_normalized
        ).reshape(-1, int(label_dim)),
        "drnl_active_attribution": np.concatenate(drnl_active).astype(np.float32) if drnl_active else np.empty(0, np.float32),
        "edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "edge_index": np.concatenate(edge_indices).reshape(-1, 2).astype(np.int64) if edge_indices else np.empty((0, 2), np.int64),
        "edge_feature_values": np.concatenate(edge_features).reshape(-1, edge_width).astype(np.float32) if edge_features else np.empty((0, edge_width), np.float32),
        "edge_role_pairs": np.concatenate(edge_roles).reshape(-1, 2).astype(np.int8) if edge_roles else np.empty((0, 2), np.int8),
        "edge_expected_gradients": np.concatenate(edge_values).reshape(-1, edge_width) if edge_values else np.empty((0, edge_width), np.float32),
        "edge_expected_gradients_abs_proportion": np.concatenate(edge_normalized).reshape(-1, edge_width) if edge_normalized else np.empty((0, edge_width), np.float32),
        "edge_baseline_fallback_tiers": np.concatenate(edge_fallbacks).astype(np.int8) if edge_fallbacks else np.empty(0, np.int8),
        "edge_baseline_seed": np.asarray(edge_seeds, dtype=np.int64),
        "edge_pool_sha256": np.asarray(
            [edge_pool.sha256 if variant == "directed_ecc" else ""] * len(test_graphs),
            dtype="U64",
        ),
    }
    for prefix, diagnostics in (
        ("node", node_diagnostics), ("drnl", drnl_diagnostics)
    ):
        for key in ("completeness_absolute", "completeness_relative", "output_difference", "attribution_sum"):
            result["{}_{}".format(prefix, key)] = _diag_array(diagnostics, key)
    if variant == "directed_ecc":
        for key in ("completeness_absolute", "completeness_relative", "output_difference", "attribution_sum"):
            result["edge_{}".format(key)] = _diag_array(edge_diagnostics, key)
    else:
        for key in ("completeness_absolute", "completeness_relative", "output_difference", "attribution_sum"):
            result["edge_{}".format(key)] = np.zeros(len(test_graphs), dtype=np.float64)
    if faithfulness_background_samples:
        result.update({
            "faithfulness_node_family_order": np.asarray(node_family_order, dtype="U64"),
            "faithfulness_node_top_family": np.asarray(faithfulness_node_top, dtype=np.int16),
            "faithfulness_node_bottom_family": np.asarray(faithfulness_node_bottom, dtype=np.int16),
            "faithfulness_node_top_abs_log_odds_change": np.asarray(faithfulness_node_top_change, dtype=np.float64),
            "faithfulness_node_bottom_abs_log_odds_change": np.asarray(faithfulness_node_bottom_change, dtype=np.float64),
            "faithfulness_edge_family_order": np.asarray(edge_family_order, dtype="U64"),
            "faithfulness_edge_top_family": np.asarray(faithfulness_edge_top if variant == "directed_ecc" else [-1] * len(test_graphs), dtype=np.int16),
            "faithfulness_edge_bottom_family": np.asarray(faithfulness_edge_bottom if variant == "directed_ecc" else [-1] * len(test_graphs), dtype=np.int16),
            "faithfulness_edge_top_abs_log_odds_change": np.asarray(faithfulness_edge_top_change if variant == "directed_ecc" else [0.0] * len(test_graphs), dtype=np.float64),
            "faithfulness_edge_bottom_abs_log_odds_change": np.asarray(faithfulness_edge_bottom_change if variant == "directed_ecc" else [0.0] * len(test_graphs), dtype=np.float64),
        })
    return result

