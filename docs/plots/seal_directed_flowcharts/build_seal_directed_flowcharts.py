"""Build four publication-ready SEAL_directed workflow diagrams.

Two diagrams document the implemented node-attributed pipeline. Two diagrams
describe an explicitly unimplemented, edge-conditioned extension inspired by
Simonovsky & Komodakis (CVPR 2017). SVG remains editable; PDF and PNG are
exported for manuscripts and presentations.

Repository sources audited for the implemented diagrams:
  - src/python/seal_directed/Main_directed.py
  - src/python/seal_directed/util_functions_directed.py
  - src/python/seal_directed/build_node_attribute_mats.py
  - src/python/pytorch_DGCNN/
  - validated 10 x 290 campaign RUN_CONFIG.json and RUN_VALIDATION.json
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent

NAVY = "#16324F"
BLUE = "#2F6F9F"
BLUE_LIGHT = "#EAF3F8"
TEAL = "#23867A"
TEAL_LIGHT = "#E6F4F1"
ORANGE = "#D9772B"
ORANGE_LIGHT = "#FDF0E5"
PURPLE = "#7257A8"
PURPLE_LIGHT = "#F0ECF8"
GREEN = "#3B7F4F"
GREEN_LIGHT = "#EAF4EC"
RED = "#B64A4A"
RED_LIGHT = "#FBECEC"
GOLD = "#A97816"
GOLD_LIGHT = "#FCF4DE"
GRAY_900 = "#23313D"
GRAY_700 = "#4F5E69"
GRAY_500 = "#7B8993"
GRAY_300 = "#CBD5DC"
GRAY_100 = "#F4F7F8"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
    }
)


def setup_canvas(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    accent,
    fill,
    number=None,
    title_size=10.5,
    body_size=8.2,
    line_spacing=1.25,
):
    ax.add_patch(
        FancyBboxPatch(
            (x + 0.05, y - 0.06),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.13",
            linewidth=0,
            facecolor="#DCE3E7",
            alpha=0.65,
            zorder=1,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.13",
            linewidth=1.15,
            edgecolor=accent,
            facecolor=fill,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (x, y + h - 0.12),
            w,
            0.12,
            linewidth=0,
            facecolor=accent,
            zorder=3,
        )
    )
    text_x = x + 0.18
    if number is not None:
        ax.text(
            x + 0.22,
            y + h - 0.40,
            str(number),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=WHITE,
            bbox=dict(boxstyle="circle,pad=0.30", facecolor=accent, edgecolor="none"),
            zorder=4,
        )
        text_x = x + 0.55
    ax.text(
        text_x,
        y + h - 0.39,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=GRAY_900,
        zorder=4,
    )
    ax.text(
        x + 0.18,
        y + h - 0.72,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=GRAY_700,
        linespacing=line_spacing,
        zorder=4,
    )


def arrow(ax, start, end, color=GRAY_500, connectionstyle="arc3", label=None, label_offset=(0, 0)):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
            zorder=5,
        )
    )
    if label:
        ax.text(
            (start[0] + end[0]) / 2 + label_offset[0],
            (start[1] + end[1]) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            color=GRAY_700,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=WHITE, edgecolor="none", alpha=0.96),
            zorder=6,
        )


def title_block(ax, title, subtitle, y, proposal=False):
    ax.text(0.55, y, title, ha="left", va="top", fontsize=22, fontweight="bold", color=NAVY)
    ax.text(0.55, y - 0.46, subtitle, ha="left", va="top", fontsize=11.2, color=GRAY_700)
    ax.plot(
        [0.55, 1.75],
        [y - 0.80, y - 0.80],
        color=RED if proposal else ORANGE,
        linewidth=4,
        solid_capstyle="round",
    )
    if proposal:
        ax.text(
            18.05,
            y - 0.10,
            "PROPOSAL · NOT IMPLEMENTED",
            ha="right",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color=RED,
            bbox=dict(boxstyle="round,pad=0.38", facecolor=RED_LIGHT, edgecolor=RED, linewidth=1),
        )


def save_all(fig, stem):
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 240})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=WHITE, **kwargs)
        print(f"[OK] {path}")
    plt.close(fig)


def lane_background(ax, y, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (0.92, y),
            17.38,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=color,
            edgecolor="none",
            alpha=0.50,
            zorder=0,
        )
    )


def lane_label(ax, y, text, color):
    ax.text(
        0.56,
        y,
        text.upper(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color=color,
    )


def protocol_strip(ax, y, items, title, color=PURPLE, fill=PURPLE_LIGHT):
    ax.add_patch(
        FancyBboxPatch(
            (0.70, y),
            17.10,
            1.20,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=fill,
            edgecolor=color,
            linewidth=0.9,
        )
    )
    ax.text(0.95, y + 0.87, title, fontsize=8.8, fontweight="bold", color=color)
    positions = [0.95, 3.20, 5.40, 7.60, 10.00, 12.25, 14.55]
    for idx, item in enumerate(items):
        ax.text(positions[idx], y + 0.39, item, fontsize=8.6, color=GRAY_700, va="center")
        if idx < len(items) - 1:
            ax.text(positions[idx + 1] - 0.25, y + 0.39, "•", fontsize=12, color=color, va="center")


def build_current_high_level():
    fig, ax = setup_canvas(18.5, 8.4, (0, 18.5), (0, 8.4))
    title_block(
        ax,
        "SEAL_directed + node attributes | High-level workflow",
        "Implemented pipeline: directional sampling and endpoint encoding with ecological node covariates",
        y=8.14,
    )
    stages = [
        (
            "Attributed food web",
            "Directed binary adjacency\n44 node attributes\nEcological role metadata",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Directional sampling",
            "Ordered resource → consumer\n90% train · 10% held out\n1:1 positives / negatives",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Enclosing subgraphs",
            "1-hop weak neighborhood\nHeld-out directed link masked\nOne graph per candidate pair",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "Node representation",
            "Distinct source/target labels\nDirected DRNL-derived tags\n+ 44 ecological attributes",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "DGCNN classifier",
            "Weak-undirected message passing\nSortPooling + 1D convolutions\nNode2Vec disabled",
            ORANGE,
            ORANGE_LIGHT,
        ),
        (
            "Prediction & evaluation",
            "P(resource → consumer)\nROC-AUC · PR-AUC · F1\nPrecision · Recall · MCC · TSS",
            PURPLE,
            PURPLE_LIGHT,
        ),
    ]
    w, h, gap = 2.55, 2.63, 0.38
    x0, y = 0.65, 3.56
    for idx, (name, body, accent, fill) in enumerate(stages, start=1):
        x = x0 + (idx - 1) * (w + gap)
        rounded_box(ax, x, y, w, h, name, body, accent, fill, number=idx, title_size=9.0, body_size=7.8)
        if idx < len(stages):
            arrow(ax, (x + w + 0.03, y + h / 2), (x + w + gap - 0.03, y + h / 2))

    protocol_strip(
        ax,
        1.28,
        [
            "290 webs",
            "10 runs/web",
            "2,900 valid runs",
            "h = 1",
            "50 epochs",
            "threshold = 0.5",
            "no edge weights",
        ],
        "VALIDATED CUDA CAMPAIGN · RAW RESULTS, NO OUTLIER FILTER",
    )
    ax.text(
        17.82,
        0.58,
        "Implemented SEAL_directed_gpu_v1 · source commit 220d0a21b7fa",
        ha="right",
        fontsize=8.2,
        color=GRAY_500,
    )
    save_all(fig, "seal_directed_attrs_flow_high_level")


def build_current_detailed():
    fig, ax = setup_canvas(18.8, 12.4, (0, 18.8), (0, 12.4))
    title_block(
        ax,
        "SEAL_directed + node attributes | Detailed workflow",
        "Repository-grounded data flow, directional encoding and DGCNN training configuration",
        y=12.14,
    )

    lane_background(ax, 8.60, 2.40, BLUE_LIGHT)
    lane_background(ax, 5.64, 2.40, TEAL_LIGHT)
    lane_background(ax, 2.68, 2.40, ORANGE_LIGHT)
    lane_label(ax, 9.80, "Data & sampling", BLUE)
    lane_label(ax, 6.84, "Subgraph & features", TEAL)
    lane_label(ax, 3.88, "Learning & evaluation", ORANGE)

    w, h = 3.75, 1.80
    xs = [1.22, 5.62, 10.02, 14.42]
    ys = [8.90, 5.94, 2.98]
    rows = [
        [
            (
                "Load attributed MAT",
                "net: directed binary adjacency\ngroup: 44-column node matrix\nrole: consumer/resource/both",
            ),
            (
                "Split positive links",
                "Keep ordered non-self links (i, j)\nShuffle with run-specific seed\nceil(0.90 × L) train; rest test",
            ),
            (
                "Sample directed negatives",
                "Unknown ordered resource → consumer pairs\nUniform without replacement; 1:1 balance\nFallback to all directed non-links if needed",
            ),
            (
                "Build observed graph A",
                "Copy binary net\nRemove held-out positive i → j links\nPreserve reverse j → i context if observed",
            ),
        ],
        [
            (
                "Candidate pair (source, target)",
                "Resource endpoint stays first\nConsumer endpoint stays second\nGraph label: interaction = 1, non-link = 0",
            ),
            (
                "Extract h = 1 neighborhood",
                "Collect incoming + outgoing neighbors\nInduce directed local adjacency\nMask source → target inside every subgraph",
            ),
            (
                "Directional structural tags",
                "Weak-projection DRNL for other nodes\nsource label = 1; target label = 2\nShift remaining positive labels by +2",
            ),
            (
                "Attach 44 node attributes",
                "2 mass features · 6 taxonomy levels\n10 metabolic · 7 movement · 19 life-stage\nSlice rows to the extracted subgraph",
            ),
        ],
        [
            (
                "Create DGCNN input",
                "One-hot structural tag ⊕ 44 attributes\nWeak-undirected NetworkX projection\nNo Node2Vec; no edge features",
            ),
            (
                "Learn graph representation",
                "Latent widths: 32, 32, 32, 1\nConcatenate convolution outputs\nSortPooling: 60th percentile, min k = 10",
            ),
            (
                "Optimise classifier",
                "1D convolutions + MLP(128), dropout\nAdam, lr = 10⁻⁴; batch size 50\n50 epochs; select minimum validation loss",
            ),
            (
                "Evaluate held-out pairs",
                "Class-1 probability is link score\nPositive if score > 0.5\nReport seven predictive metrics per run",
            ),
        ],
    ]
    accents = [BLUE, TEAL, ORANGE]
    for row_idx, (row, y, accent) in enumerate(zip(rows, ys, accents)):
        row_xs = xs if row_idx != 1 else list(reversed(xs))
        for idx, (name, body) in enumerate(row):
            rounded_box(
                ax,
                row_xs[idx],
                y,
                w,
                h,
                name,
                body,
                accent,
                WHITE,
                number=row_idx * 4 + idx + 1,
                title_size=9.8,
                body_size=7.6,
                line_spacing=1.20,
            )
    for y in (ys[0], ys[2]):
        for idx in range(3):
            arrow(ax, (xs[idx] + w + 0.04, y + h / 2), (xs[idx + 1] - 0.04, y + h / 2))
    for idx in range(3, 0, -1):
        arrow(ax, (xs[idx] - 0.04, ys[1] + h / 2), (xs[idx - 1] + w + 0.04, ys[1] + h / 2))
    arrow(
        ax,
        (xs[3] + w / 2, ys[0] - 0.04),
        (xs[3] + w / 2, ys[1] + h + 0.04),
        label="each sampled ordered pair",
        label_offset=(1.25, 0),
    )
    arrow(
        ax,
        (xs[0] + w / 2, ys[1] - 0.04),
        (xs[0] + w / 2, ys[2] + h + 0.04),
        label="encoded graph examples",
        label_offset=(1.18, 0),
    )

    ax.add_patch(
        FancyBboxPatch(
            (1.22, 1.30),
            16.95,
            0.94,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=PURPLE_LIGHT,
            edgecolor=PURPLE,
            linewidth=0.9,
        )
    )
    ax.text(1.48, 1.94, "IMPLEMENTATION BOUNDARY", fontsize=8.7, fontweight="bold", color=PURPLE)
    ax.text(
        4.35,
        1.94,
        "Direction affects ordered sampling, target masking and endpoint labels; the current DGCNN aggregates on a weak-undirected projection.",
        fontsize=8.7,
        color=GRAY_700,
        va="center",
    )
    ax.text(
        1.48,
        1.57,
        "The role vector constrains negative candidates; it is not concatenated to the learned 44-feature node representation. No link weights are used.",
        fontsize=8.6,
        color=GRAY_700,
        va="center",
    )
    ax.text(
        18.15,
        0.67,
        "Sources: Main_directed.py · util_functions_directed.py · build_node_attribute_mats.py · pytorch_DGCNN",
        ha="right",
        fontsize=8.1,
        color=GRAY_500,
    )
    save_all(fig, "seal_directed_attrs_flow_detailed")


def build_proposal_high_level():
    fig, ax = setup_canvas(18.5, 8.6, (0, 18.5), (0, 8.6))
    title_block(
        ax,
        "Edge-conditioned SEAL_directed | High-level proposal",
        "Add leakage-safe context-edge covariates while retaining the current node-attributed SEAL framework",
        y=8.34,
        proposal=True,
    )
    stages = [
        (
            "Current inputs",
            "Directed binary food web\n44 node attributes\nRole-aware candidate sampling",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Context-edge covariates",
            "Start: log mass ratio\n(consumer / resource)\n+ missingness indicator\nLater: interaction descriptors",
            GOLD,
            GOLD_LIGHT,
        ),
        (
            "Leakage-safe subgraphs",
            "Same ordered pairs; h = 1\nRemove target edge + attributes\nFit transforms on training only",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "Edge-conditioned messages",
            "ECC-style filter/gate from eᵢⱼ\nIncoming/outgoing channels\nNode tags + 44 attributes retained",
            RED,
            RED_LIGHT,
        ),
        (
            "Graph classification",
            "DGCNN-style pooling and classifier\nPredict P(resource → consumer)\nNo target-edge attribute supplied",
            ORANGE,
            ORANGE_LIGHT,
        ),
        (
            "Controlled evaluation",
            "Fixed splits · seeds · negatives\nStructure / node / edge ablations\nPerformance + robustness",
            PURPLE,
            PURPLE_LIGHT,
        ),
    ]
    w, h, gap = 2.55, 2.74, 0.38
    x0, y = 0.65, 3.62
    for idx, (name, body, accent, fill) in enumerate(stages, start=1):
        x = x0 + (idx - 1) * (w + gap)
        rounded_box(ax, x, y, w, h, name, body, accent, fill, number=idx, title_size=8.25, body_size=7.2)
        if idx < len(stages):
            arrow(ax, (x + w + 0.03, y + h / 2), (x + w + gap - 0.03, y + h / 2))

    ax.add_patch(
        FancyBboxPatch(
            (0.70, 1.18),
            17.10,
            1.34,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=GRAY_100,
            edgecolor=GRAY_300,
            linewidth=1,
        )
    )
    ax.text(0.96, 2.17, "SCIENTIFIC INTERPRETATION", fontsize=8.9, fontweight="bold", color=NAVY)
    ax.text(
        0.96,
        1.76,
        "The initial mass-ratio channel is a derived edge covariate—not measured interaction strength. It adds a relational inductive bias,",
        fontsize=8.8,
        color=GRAY_700,
    )
    ax.text(
        0.96,
        1.43,
        "but is partly redundant with node mass. True interaction weights can use the same architecture only when independently observed.",
        fontsize=8.8,
        color=GRAY_700,
    )
    ax.text(
        17.82,
        0.55,
        "Primary methodological basis: Simonovsky & Komodakis, CVPR 2017 (ECC)",
        ha="right",
        fontsize=8.2,
        color=GRAY_500,
    )
    save_all(fig, "seal_directed_edge_weight_proposal_high_level")


def build_proposal_detailed():
    fig, ax = setup_canvas(18.8, 15.2, (0, 18.8), (0, 15.2))
    title_block(
        ax,
        "Edge-conditioned SEAL_directed | Detailed proposal",
        "A staged, leakage-safe route from relational edge covariates to edge-conditioned graph classification",
        y=14.94,
        proposal=True,
    )

    lane_ys = [11.38, 8.42, 5.46, 2.50]
    lane_colors = [GOLD_LIGHT, TEAL_LIGHT, RED_LIGHT, PURPLE_LIGHT]
    lane_names = [
        ("Edge data", GOLD),
        ("Subgraph assembly", TEAL),
        ("Edge-conditioned learning", RED),
        ("Experiment & safeguards", PURPLE),
    ]
    for y, fill, (name, color) in zip(lane_ys, lane_colors, lane_names):
        lane_background(ax, y, 2.40, fill)
        lane_label(ax, y + 1.20, name, color)

    w, h = 3.75, 1.80
    xs = [1.22, 5.62, 10.02, 14.42]
    rows = [
        [
            (
                "Define context edges",
                "Only observed training-graph links carry eᵢⱼ\nTarget candidate edge is absent\nKeep resource → consumer orientation",
            ),
            (
                "Phase 1 edge vector",
                "rᵢⱼ = log₁₀(m_consumer / m_resource)\nMass-missing indicator\nOptional explicit direction channel",
            ),
            (
                "Phase 2 enrichment",
                "One-hot interaction type\nDimensionality and classification\nUse only fields available at inference time",
            ),
            (
                "Train-only preprocessing",
                "Fit imputation and scaling on training webs\nTransform validation/test without refitting\nRecord provenance and missingness",
            ),
        ],
        [
            (
                "Reuse current candidates",
                "Same positive splits, role-aware negatives\nSame seeds and 1:1 class balance\nNo protocol change during ablation",
            ),
            (
                "Mask the target interaction",
                "Remove source → target from adjacency\nRemove its edge vector simultaneously\nNever expose held-out interaction metadata",
            ),
            (
                "Extract directed h = 1 context",
                "Preserve ordered endpoints and local directions\nAttach eᵢⱼ to observed context edges\nRetain 44 node attributes + node tags",
            ),
            (
                "Construct graph tensors",
                "Node tensor: structural tag ⊕ attributes\nEdge tensor: covariates + missingness\nDirected edge index or paired direction channels",
            ),
        ],
        [
            (
                "Edge network / gate",
                "Fθ(eⱼᵢ) generates an edge-specific filter\nor gθ(eⱼᵢ) gates the neighbor message\nShared parameters across all subgraphs",
            ),
            (
                "Directional aggregation",
                "hᵢˡ = σ[Σⱼ Fθ(eⱼᵢ) hⱼˡ⁻¹ + W₀hᵢˡ⁻¹]\nSeparate incoming/outgoing transforms\nNormalize by neighborhood size",
            ),
            (
                "Pool candidate subgraph",
                "Concatenate message-passing layers\nUse SortPooling or attention pooling\nKeep ordered endpoint identity visible",
            ),
            (
                "Classify candidate link",
                "MLP → P(resource → consumer)\nBinary loss as in current SEAL_directed\nOptionally calibrate probabilities post hoc",
            ),
        ],
        [
            (
                "Frozen ablation ladder",
                "A: structure only\nB: + 44 node attributes\nC: + mass-ratio edge channel\nD: + richer interaction descriptors",
            ),
            (
                "Fair comparison",
                "Reuse splits, seeds and negative samples\nMatch training budget and threshold policy\nReport paired per-web differences",
            ),
            (
                "Leakage & missingness audit",
                "Assert target-edge data are absent\nReport feature coverage and fallback strata\nSensitivity analysis with missingness masks",
            ),
            (
                "Decision gate",
                "Retain extension only if gains are robust\nSeparate existence from strength prediction\nDocument compute cost and uncertainty",
            ),
        ],
    ]
    accents = [GOLD, TEAL, RED, PURPLE]
    for row_idx, (row, y, accent) in enumerate(zip(rows, lane_ys, accents)):
        row_xs = xs if row_idx % 2 == 0 else list(reversed(xs))
        for idx, (name, body) in enumerate(row):
            rounded_box(
                ax,
                row_xs[idx],
                y + 0.30,
                w,
                h,
                name,
                body,
                accent,
                WHITE,
                number=row_idx * 4 + idx + 1,
                title_size=9.5,
                body_size=7.35,
                line_spacing=1.18,
            )

    # Snake flow within each lane.
    for row_idx, y in enumerate(lane_ys):
        yy = y + 0.30 + h / 2
        if row_idx % 2 == 0:
            for idx in range(3):
                arrow(ax, (xs[idx] + w + 0.04, yy), (xs[idx + 1] - 0.04, yy))
        else:
            for idx in range(3, 0, -1):
                arrow(ax, (xs[idx] - 0.04, yy), (xs[idx - 1] + w + 0.04, yy))

    arrow(
        ax,
        (xs[3] + w / 2, lane_ys[0] + 0.26),
        (xs[3] + w / 2, lane_ys[1] + 0.30 + h + 0.04),
        label="edge schema frozen",
        label_offset=(1.03, 0),
    )
    arrow(
        ax,
        (xs[0] + w / 2, lane_ys[1] + 0.26),
        (xs[0] + w / 2, lane_ys[2] + 0.30 + h + 0.04),
        label="graph tensors",
        label_offset=(0.88, 0),
    )
    arrow(
        ax,
        (xs[3] + w / 2, lane_ys[2] + 0.26),
        (xs[3] + w / 2, lane_ys[3] + 0.30 + h + 0.04),
        label="paired evaluation",
        label_offset=(1.00, 0),
    )

    ax.add_patch(
        FancyBboxPatch(
            (1.22, 1.06),
            16.95,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=GRAY_100,
            edgecolor=GRAY_300,
            linewidth=0.9,
        )
    )
    ax.text(1.48, 1.67, "IMPLEMENTATION ORDER", fontsize=8.7, fontweight="bold", color=NAVY)
    ax.text(
        4.00,
        1.67,
        "1) scalar edge gate with mass ratio → 2) true ECC-style vector filters → 3) richer descriptors only after coverage and leakage audits.",
        fontsize=8.7,
        color=GRAY_700,
        va="center",
    )
    ax.text(
        1.48,
        1.33,
        "This design is a research proposal, not a description of the current repository. The present model remains binary-edge and weak-undirected inside DGCNN.",
        fontsize=8.6,
        color=RED,
        va="center",
    )
    ax.text(
        18.15,
        0.53,
        "Method basis: Dynamic Edge-Conditioned Filters (Simonovsky & Komodakis, CVPR 2017)",
        ha="right",
        fontsize=8.1,
        color=GRAY_500,
    )
    save_all(fig, "seal_directed_edge_weight_proposal_detailed")


if __name__ == "__main__":
    build_current_high_level()
    build_current_detailed()
    build_proposal_high_level()
    build_proposal_detailed()
