"""Expected Gradients for 44 node attributes and separate DRNL Integrated Gradients."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


CONFUSION_CODES = {"TN": 0, "FP": 1, "FN": 2, "TP": 3}
NODE_ROLE_CODES = {"source_resource": 0, "target_consumer": 1, "context": 2}


def one_hot_drnl(graph, width, device):
    labels = torch.as_tensor(graph.node_tags, dtype=torch.long, device=device)
    if labels.numel() != int(graph.num_nodes):
        raise ValueError("DRNL label count does not match graph nodes")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= int(width)):
        raise ValueError("DRNL label is outside checkpoint feat_dim")
    one_hot = torch.zeros((labels.numel(), int(width)), dtype=torch.float32, device=device)
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    return one_hot


class AttributeLogOdds(nn.Module):
    def __init__(self, classifier, graph, fixed_drnl):
        super().__init__()
        self.classifier = classifier
        self.graph = graph
        self.register_buffer("fixed_drnl", fixed_drnl.detach().clone())

    def forward(self, node_attributes):
        if node_attributes.ndim == 2:
            node_attributes = node_attributes.unsqueeze(0)
        if node_attributes.ndim != 3:
            raise ValueError("Expected [batch, nodes, attributes]")
        batch = int(node_attributes.shape[0])
        tags = self.fixed_drnl.unsqueeze(0).expand(batch, -1, -1).reshape(
            -1, self.fixed_drnl.shape[1]
        )
        attributes = node_attributes.reshape(-1, node_attributes.shape[-1])
        return self.classifier.forward_log_odds(
            [self.graph] * batch, tags, node_attributes=attributes
        )


class DrnlLogOdds(nn.Module):
    def __init__(self, classifier, graph, fixed_attributes):
        super().__init__()
        self.classifier = classifier
        self.graph = graph
        self.register_buffer("fixed_attributes", fixed_attributes.detach().clone())

    def forward(self, drnl_onehot):
        if drnl_onehot.ndim == 2:
            drnl_onehot = drnl_onehot.unsqueeze(0)
        if drnl_onehot.ndim != 3:
            raise ValueError("Expected [batch, nodes, DRNL channels]")
        batch = int(drnl_onehot.shape[0])
        tags = drnl_onehot.reshape(-1, drnl_onehot.shape[-1])
        attributes = self.fixed_attributes.unsqueeze(0).expand(batch, -1, -1).reshape(
            -1, self.fixed_attributes.shape[1]
        )
        return self.classifier.forward_log_odds(
            [self.graph] * batch, tags, node_attributes=attributes
        )


def build_training_backgrounds(train_graphs, query_nodes, samples, rng):
    """Create role-matched empirical baselines using training subgraphs only."""
    if samples < 1 or not train_graphs:
        raise ValueError("Expected Gradients requires at least one training background")
    required_context = int(query_nodes) - 2
    eligible = [
        index for index, graph in enumerate(train_graphs)
        if required_context == 0 or int(graph.num_nodes) > 2
    ]
    if not eligible:
        raise ValueError("No training subgraph contains context nodes for this candidate")
    reference_indices = rng.choice(eligible, size=int(samples), replace=True)
    feature_width = int(np.asarray(train_graphs[eligible[0]].node_features).shape[1])
    baselines = np.empty((int(samples), int(query_nodes), feature_width), dtype=np.float32)
    for sample_index, graph_index in enumerate(reference_indices):
        reference = np.asarray(train_graphs[int(graph_index)].node_features, dtype=np.float32)
        baselines[sample_index, 0] = reference[0]
        baselines[sample_index, 1] = reference[1]
        if required_context:
            context = reference[2:]
            if context.shape[0] == 0:
                raise AssertionError("Eligible training background has no context")
            if context.shape[0] >= required_context:
                selected = rng.choice(
                    context.shape[0], size=required_context, replace=False
                )
            else:
                # Retain every contextual node from the reference subgraph,
                # then complete only the missing rows by sampling from that
                # same training context with replacement.
                selected = np.concatenate([
                    np.arange(context.shape[0], dtype=np.int64),
                    rng.choice(
                        context.shape[0],
                        size=required_context - context.shape[0],
                        replace=True,
                    ),
                ])
            baselines[sample_index, 2:] = context[selected]
    return baselines, np.asarray(reference_indices, dtype=np.int64)


def expected_gradients(
    wrapper,
    inputs,
    baselines,
    rng,
    internal_batch_size=32,
):
    """Monte Carlo Expected Gradients with empirical baselines and U(0,1) paths."""
    if inputs.ndim != 2 or baselines.ndim != 3 or baselines.shape[1:] != inputs.shape:
        raise ValueError("Expected Gradients input/baseline shape mismatch")
    samples = int(baselines.shape[0])
    internal_batch_size = max(1, int(internal_batch_size))
    alpha = torch.as_tensor(
        rng.random((samples, 1, 1)), dtype=inputs.dtype, device=inputs.device
    )
    baseline_tensor = torch.as_tensor(baselines, dtype=inputs.dtype, device=inputs.device)
    total = torch.zeros_like(inputs)
    baseline_outputs = []
    for start in range(0, samples, internal_batch_size):
        stop = min(start + internal_batch_size, samples)
        current_baselines = baseline_tensor[start:stop]
        interpolated = (
            current_baselines + alpha[start:stop] * (inputs.unsqueeze(0) - current_baselines)
        ).detach().requires_grad_(True)
        outputs = wrapper(interpolated)
        gradients = torch.autograd.grad(outputs.sum(), interpolated, create_graph=False)[0]
        if not torch.all(torch.isfinite(gradients)):
            raise ValueError("Non-finite Expected Gradients derivative")
        total += torch.sum((inputs.unsqueeze(0) - current_baselines) * gradients, dim=0)
        with torch.no_grad():
            baseline_outputs.append(wrapper(current_baselines).detach())
    attributions = total / float(samples)
    with torch.no_grad():
        input_output = wrapper(inputs.unsqueeze(0))[0]
        mean_baseline_output = torch.cat(baseline_outputs).mean()
    output_difference = input_output - mean_baseline_output
    attribution_sum = attributions.sum()
    absolute_error = torch.abs(attribution_sum - output_difference)
    relative_error = absolute_error / torch.clamp(torch.abs(output_difference), min=1e-12)
    return attributions, {
        "input_output": float(input_output.detach().cpu()),
        "mean_baseline_output": float(mean_baseline_output.detach().cpu()),
        "output_difference": float(output_difference.detach().cpu()),
        "attribution_sum": float(attribution_sum.detach().cpu()),
        "completeness_absolute": float(absolute_error.detach().cpu()),
        "completeness_relative": float(relative_error.detach().cpu()),
    }


def integrated_gradients_drnl(
    wrapper,
    drnl_onehot,
    steps,
    internal_batch_size=32,
):
    try:
        from captum.attr import IntegratedGradients
    except ImportError as error:
        raise RuntimeError("captum==0.9.0 is required for DRNL Integrated Gradients") from error
    baseline = torch.zeros_like(drnl_onehot).unsqueeze(0)
    inputs = drnl_onehot.unsqueeze(0)
    method = IntegratedGradients(wrapper)
    attributions, _ = method.attribute(
        inputs,
        baselines=baseline,
        n_steps=int(steps),
        method="gausslegendre",
        internal_batch_size=max(1, int(internal_batch_size)),
        return_convergence_delta=True,
    )
    attributions = attributions[0]
    with torch.no_grad():
        input_output = wrapper(inputs)[0]
        baseline_output = wrapper(baseline)[0]
    output_difference = input_output - baseline_output
    attribution_sum = attributions.sum()
    absolute_error = torch.abs(attribution_sum - output_difference)
    relative_error = absolute_error / torch.clamp(torch.abs(output_difference), min=1e-12)
    if not torch.all(torch.isfinite(attributions)):
        raise ValueError("Non-finite DRNL Integrated Gradients attribution")
    return attributions, {
        "input_output": float(input_output.detach().cpu()),
        "baseline_output": float(baseline_output.detach().cpu()),
        "output_difference": float(output_difference.detach().cpu()),
        "attribution_sum": float(attribution_sum.detach().cpu()),
        "completeness_absolute": float(absolute_error.detach().cpu()),
        "completeness_relative": float(relative_error.detach().cpu()),
    }


def _prediction(classifier, graph, tags, attributes):
    with torch.no_grad():
        log_probabilities = classifier.forward_prepared(
            [graph], tags, node_attributes=attributes, labels=None
        )[0]
    return (
        float(torch.exp(log_probabilities[1]).detach().cpu()),
        float((log_probabilities[1] - log_probabilities[0]).detach().cpu()),
    )


def _confusion_code(label, score, threshold):
    predicted = int(float(score) > float(threshold))
    if int(label) == 1:
        return CONFUSION_CODES["TP" if predicted else "FN"]
    return CONFUSION_CODES["FP" if predicted else "TN"]


def attribute_test_graphs(
    classifier,
    train_graphs,
    test_graphs,
    expected_scores,
    feat_dim,
    attribution_seed,
    expected_gradients_samples,
    drnl_steps,
    threshold=0.5,
    internal_batch_size=32,
    score_tolerance=1e-7,
    feature_families=None,
    faithfulness_background_samples=0,
    scores_prevalidated=False,
):
    classifier.eval()
    device = next(classifier.parameters()).device
    if expected_scores is not None:
        expected_scores = np.asarray(expected_scores, dtype=np.float64).reshape(-1)
        if expected_scores.size != len(test_graphs):
            raise ValueError("Stored score count does not match test graphs")
    if faithfulness_background_samples:
        if feature_families is None or len(feature_families) != 44:
            raise ValueError("Faithfulness validation requires the 44 feature-family labels")

    node_offsets = [0]
    node_ids = []
    node_roles = []
    drnl_labels = []
    eg_values = []
    eg_normalized = []
    drnl_values = []
    drnl_active = []
    candidate_pairs = []
    candidate_labels = []
    candidate_scores = []
    candidate_log_odds = []
    candidate_confusion = []
    background_indices = []
    eg_diagnostics = []
    drnl_diagnostics = []
    faithfulness_top_family = []
    faithfulness_bottom_family = []
    faithfulness_top_reduction = []
    faithfulness_bottom_reduction = []

    for candidate_index, graph in enumerate(test_graphs):
        candidate_rng = np.random.default_rng(
            int(attribution_seed) + 1_000_003 * int(candidate_index)
        )
        tags = one_hot_drnl(graph, feat_dim, device)
        attributes = torch.as_tensor(
            np.asarray(graph.node_features, dtype=np.float32), device=device
        )
        score, log_odds = _prediction(classifier, graph, tags, attributes)
        if expected_scores is not None and not scores_prevalidated and not math.isclose(
            score,
            float(expected_scores[candidate_index]),
            rel_tol=0.0,
            abs_tol=score_tolerance,
        ):
            raise ValueError(
                "Checkpoint score mismatch for candidate {}: stored {:.12g}, reconstructed {:.12g}".format(
                    candidate_index, expected_scores[candidate_index], score
                )
            )
        baselines, selected_backgrounds = build_training_backgrounds(
            train_graphs,
            graph.num_nodes,
            expected_gradients_samples,
            candidate_rng,
        )
        eg, eg_diag = expected_gradients(
            AttributeLogOdds(classifier, graph, tags),
            attributes,
            baselines,
            candidate_rng,
            internal_batch_size=internal_batch_size,
        )
        drnl, drnl_diag = integrated_gradients_drnl(
            DrnlLogOdds(classifier, graph, attributes),
            tags,
            drnl_steps,
            internal_batch_size=internal_batch_size,
        )
        eg_np = eg.detach().cpu().numpy().astype(np.float32)
        denominator = float(np.sum(np.abs(eg_np)))
        eg_norm = np.abs(eg_np) / denominator if denominator > 0 else np.zeros_like(eg_np)
        drnl_np = drnl.detach().cpu().numpy().astype(np.float32)
        labels = np.asarray(graph.node_tags, dtype=np.int64)
        active = drnl_np[np.arange(labels.size), labels]

        if faithfulness_background_samples:
            family_order = list(dict.fromkeys(feature_families))
            family_importance = []
            for family in family_order:
                columns = np.asarray(
                    [name == family for name in feature_families], dtype=bool
                )
                family_importance.append(float(np.sum(np.abs(eg_np[:, columns]))))
            top_index = int(np.argmax(family_importance))
            bottom_index = int(np.argmin(family_importance))
            top_family = family_order[top_index]
            bottom_family = family_order[bottom_index]
            sample_count = min(int(faithfulness_background_samples), baselines.shape[0])
            empirical = torch.as_tensor(
                baselines[:sample_count], dtype=attributes.dtype, device=device
            )
            original = attributes.unsqueeze(0).expand(sample_count, -1, -1)
            reductions = []
            for family in (top_family, bottom_family):
                columns = torch.as_tensor(
                    [name == family for name in feature_families],
                    dtype=torch.bool,
                    device=device,
                )
                perturbed = original.clone()
                perturbed[:, :, columns] = empirical[:, :, columns]
                with torch.no_grad():
                    perturbed_log_odds = AttributeLogOdds(classifier, graph, tags)(perturbed)
                reductions.append(float(np.median(log_odds - perturbed_log_odds.cpu().numpy())))
            faithfulness_top_family.append(top_index)
            faithfulness_bottom_family.append(bottom_index)
            faithfulness_top_reduction.append(reductions[0])
            faithfulness_bottom_reduction.append(reductions[1])

        candidate_pairs.append(np.asarray(graph.candidate_pair, dtype=np.int64))
        candidate_labels.append(int(graph.label))
        candidate_scores.append(
            float(expected_scores[candidate_index]) if expected_scores is not None else score
        )
        candidate_log_odds.append(log_odds)
        candidate_confusion.append(_confusion_code(
            graph.label,
            float(expected_scores[candidate_index]) if expected_scores is not None else score,
            threshold,
        ))
        node_ids.append(np.asarray(graph.node_ids, dtype=np.int64))
        roles = np.full(int(graph.num_nodes), NODE_ROLE_CODES["context"], dtype=np.int8)
        roles[0] = NODE_ROLE_CODES["source_resource"]
        roles[1] = NODE_ROLE_CODES["target_consumer"]
        node_roles.append(roles)
        drnl_labels.append(labels)
        eg_values.append(eg_np)
        eg_normalized.append(eg_norm.astype(np.float32))
        drnl_values.append(drnl_np)
        drnl_active.append(active.astype(np.float32))
        background_indices.append(selected_backgrounds)
        eg_diagnostics.append(eg_diag)
        drnl_diagnostics.append(drnl_diag)
        node_offsets.append(node_offsets[-1] + int(graph.num_nodes))

    def diagnostic(name, diagnostics):
        return np.asarray([item[name] for item in diagnostics], dtype=np.float64)

    result = {
        "candidate_pairs": np.stack(candidate_pairs).astype(np.int64),
        "candidate_labels": np.asarray(candidate_labels, dtype=np.int8),
        "candidate_scores": np.asarray(candidate_scores, dtype=np.float64),
        "candidate_log_odds": np.asarray(candidate_log_odds, dtype=np.float64),
        "candidate_confusion": np.asarray(candidate_confusion, dtype=np.int8),
        "node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "node_ids": np.concatenate(node_ids).astype(np.int64),
        "node_roles": np.concatenate(node_roles).astype(np.int8),
        "expected_gradients": np.concatenate(eg_values, axis=0),
        "expected_gradients_abs_proportion": np.concatenate(eg_normalized, axis=0),
        "drnl_labels": np.concatenate(drnl_labels).astype(np.int64),
        "drnl_integrated_gradients": np.concatenate(drnl_values, axis=0),
        "drnl_active_attribution": np.concatenate(drnl_active).astype(np.float32),
        "background_train_indices": np.stack(background_indices).astype(np.int64),
        "eg_completeness_absolute": diagnostic("completeness_absolute", eg_diagnostics),
        "eg_completeness_relative": diagnostic("completeness_relative", eg_diagnostics),
        "eg_output_difference": diagnostic("output_difference", eg_diagnostics),
        "eg_attribution_sum": diagnostic("attribution_sum", eg_diagnostics),
        "drnl_completeness_absolute": diagnostic("completeness_absolute", drnl_diagnostics),
        "drnl_completeness_relative": diagnostic("completeness_relative", drnl_diagnostics),
        "drnl_output_difference": diagnostic("output_difference", drnl_diagnostics),
        "drnl_attribution_sum": diagnostic("attribution_sum", drnl_diagnostics),
    }
    if faithfulness_background_samples:
        result.update({
            "faithfulness_family_order": np.asarray(family_order, dtype="U32"),
            "faithfulness_top_family": np.asarray(faithfulness_top_family, dtype=np.int8),
            "faithfulness_bottom_family": np.asarray(faithfulness_bottom_family, dtype=np.int8),
            "faithfulness_top_log_odds_reduction": np.asarray(
                faithfulness_top_reduction, dtype=np.float64
            ),
            "faithfulness_bottom_log_odds_reduction": np.asarray(
                faithfulness_bottom_reduction, dtype=np.float64
            ),
        })
    return result
