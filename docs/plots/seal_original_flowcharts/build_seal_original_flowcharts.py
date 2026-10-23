"""Build publication-ready flow diagrams for the repository's SEAL_original pipeline.

The content is grounded in:
  - src/python/Main.py
  - src/python/util_functions.py
  - src/python/pytorch_DGCNN/
  - src/python/run_foodweb_seal.py

Run with the Foodweb environment. SVG remains editable; PDF and PNG are
exported for manuscript and presentation use.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent

NAVY = "#16324F"
BLUE = "#2F6F9F"
BLUE_LIGHT = "#EAF3F8"
TEAL = "#2A8C82"
TEAL_LIGHT = "#E8F5F2"
ORANGE = "#D9772B"
ORANGE_LIGHT = "#FDF0E5"
PURPLE = "#7257A8"
PURPLE_LIGHT = "#F0ECF8"
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
    title_size=11,
    body_size=9,
    line_spacing=1.3,
):
    shadow = FancyBboxPatch(
        (x + 0.05, y - 0.06),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        linewidth=0,
        facecolor="#DCE3E7",
        alpha=0.65,
        zorder=1,
    )
    ax.add_patch(shadow)
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        linewidth=1.15,
        edgecolor=accent,
        facecolor=fill,
        zorder=2,
    )
    ax.add_patch(box)
    ax.add_patch(Rectangle((x, y + h - 0.12), w, 0.12, linewidth=0, facecolor=accent, zorder=3))

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
    return box


def arrow(ax, start, end, color=GRAY_500, connectionstyle="arc3", label=None, label_offset=(0, 0)):
    patch = FancyArrowPatch(
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
    ax.add_patch(patch)
    if label:
        mid_x = (start[0] + end[0]) / 2 + label_offset[0]
        mid_y = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mid_x,
            mid_y,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=GRAY_700,
            bbox=dict(boxstyle="round,pad=0.18", facecolor=WHITE, edgecolor="none", alpha=0.95),
            zorder=6,
        )


def title_block(ax, title, subtitle, x=0.5, y=8.15):
    ax.text(x, y, title, ha="left", va="top", fontsize=23, fontweight="bold", color=NAVY)
    ax.text(x, y - 0.48, subtitle, ha="left", va="top", fontsize=11.5, color=GRAY_700)
    ax.plot([x, x + 1.2], [y - 0.82, y - 0.82], color=ORANGE, linewidth=4, solid_capstyle="round")


def save_all(fig, stem):
    for suffix, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": 240}),
    ):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=WHITE, **kwargs)
        print(f"[OK] {path}")
    plt.close(fig)


def build_high_level():
    fig, ax = setup_canvas(18.5, 8.4, (0, 18.5), (0, 8.4))
    title_block(
        ax,
        "SEAL_original | High-level workflow",
        "From an empirical food web to a probability for each candidate interaction",
        y=8.15,
    )

    stages = [
        (
            "Empirical food web",
            "Adjacency matrix\nfor one food web",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Link sampling",
            "90% training links\n10% held-out links\nBalanced positives / negatives",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Enclosing subgraphs",
            "1-hop neighborhood\nfor every candidate pair\nTarget link removed",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "Structural encoding",
            "DRNL node labels\nTopology only\nNo ecological attributes",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "DGCNN classifier",
            "Graph convolutions\nSortPooling + 1D convolutions\nGPU-enabled training",
            ORANGE,
            ORANGE_LIGHT,
        ),
        (
            "Prediction & evaluation",
            "Interaction probability\nROC-AUC · PR-AUC · F1\nPrecision · Recall · MCC · TSS",
            PURPLE,
            PURPLE_LIGHT,
        ),
    ]

    w, h, gap = 2.55, 2.55, 0.38
    x0, y = 0.65, 3.65
    boxes = []
    for idx, (title, body, accent, fill) in enumerate(stages, start=1):
        x = x0 + (idx - 1) * (w + gap)
        boxes.append(
            rounded_box(
                ax,
                x,
                y,
                w,
                h,
                title,
                body,
                accent,
                fill,
                number=idx,
                title_size=9.2,
                body_size=8.0,
            )
        )

    for idx in range(len(boxes) - 1):
        left_x = x0 + idx * (w + gap)
        right_x = x0 + (idx + 1) * (w + gap)
        arrow(ax, (left_x + w + 0.03, y + h / 2), (right_x - 0.03, y + h / 2))

    # Compact protocol strip: current validated experiment, kept separate from the conceptual flow.
    strip_y = 1.38
    ax.add_patch(
        FancyBboxPatch(
            (0.65, strip_y),
            17.20,
            1.20,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=GRAY_100,
            edgecolor=GRAY_300,
            linewidth=1,
        )
    )
    ax.text(0.92, strip_y + 0.86, "CURRENT VALIDATED GPU CAMPAIGN", fontsize=9, fontweight="bold", color=NAVY)
    campaign = [
        "290 food webs",
        "10 runs per web",
        "Seeds 12345–12354",
        "50 epochs",
        "Threshold 0.5",
        "Historical undirected logic preserved",
    ]
    start_x = 0.92
    widths = [1.75, 1.85, 2.05, 1.35, 1.45, 3.0]
    for idx, (text, width) in enumerate(zip(campaign, widths)):
        ax.text(start_x, strip_y + 0.40, text, fontsize=9.2, color=GRAY_700, va="center")
        start_x += width
        if idx < len(campaign) - 1:
            ax.text(start_x - 0.22, strip_y + 0.40, "•", fontsize=12, color=ORANGE, va="center")

    ax.text(
        17.85,
        0.62,
        "Repository-grounded diagram · SEAL_original",
        ha="right",
        va="center",
        fontsize=8.5,
        color=GRAY_500,
    )
    save_all(fig, "seal_original_flow_high_level")


def lane_label(ax, y, text, color):
    ax.text(
        0.55,
        y,
        text.upper(),
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=color,
        alpha=0.95,
    )


def lane_background(ax, y, h, color):
    ax.add_patch(
        FancyBboxPatch(
            (0.88, y),
            17.45,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=color,
            edgecolor="none",
            alpha=0.48,
            zorder=0,
        )
    )


def build_detailed():
    fig, ax = setup_canvas(18.8, 12.2, (0, 18.8), (0, 12.2))
    title_block(
        ax,
        "SEAL_original | Detailed computational workflow",
        "Exact historical sampling and learning logic used by the CUDA-enabled implementation",
        y=11.95,
    )

    lane_background(ax, 8.42, 2.32, BLUE_LIGHT)
    lane_background(ax, 5.55, 2.32, TEAL_LIGHT)
    lane_background(ax, 2.68, 2.32, ORANGE_LIGHT)
    lane_label(ax, 9.58, "Data & sampling", BLUE)
    lane_label(ax, 6.71, "Subgraph encoding", TEAL)
    lane_label(ax, 3.84, "Learning & evaluation", ORANGE)

    w, h = 3.75, 1.72
    xs = [1.20, 5.60, 10.00, 14.40]
    row1_y, row2_y, row3_y = 8.72, 5.85, 2.98

    row1 = [
        (
            "Load network",
            "MAT file → sparse adjacency net\nOne empirical food web per run\nSeed controls Python, NumPy and PyTorch",
        ),
        (
            "Split observed links",
            "Use triu(net, k=1)\nShuffle positive links\nceil(0.90 × L) train; remainder test",
        ),
        (
            "Sample unknown pairs",
            "random.randint over i < j and net[i,j] = 0\nMatch positive counts → 1:1 classes\nHistorical sampler is preserved unchanged",
        ),
        (
            "Build observed graph A",
            "Copy net\nMask every held-out positive link\nin both directions; eliminate zeros",
        ),
    ]
    row2 = [
        (
            "Candidate pair (u, v)",
            "Process train+/train− and test+/test−\nGraph target label: link = 1, non-link = 0\nSubgraphs extracted in parallel on CPU",
        ),
        (
            "Extract 1-hop subgraph",
            "Collect neighbors around u and v\nMove target nodes to positions 0 and 1\nInduce a NetworkX graph from A",
        ),
        (
            "Apply structural encoding",
            "Compute DRNL labels from shortest paths\nRemove the target edge inside the subgraph\nNo Node2Vec; no ecological node attributes",
        ),
        (
            "Create graph datasets",
            "One-hot DRNL node tags in GNNGraph\nBalanced positive/negative examples\n10% of training graphs reserved for validation",
        ),
    ]
    row3 = [
        (
            "DGCNN representation",
            "Graph-convolution widths: 32, 32, 32, 1\nConcatenate latent layers\nSortPooling: 60th percentile, minimum k = 10",
        ),
        (
            "Neural classifier",
            "1D convolution → max-pool → 1D convolution\nMLP hidden size 128 with dropout\nTwo-class log-softmax output",
        ),
        (
            "Optimise & select epoch",
            "Adam, learning rate 10⁻⁴; batch size 50\n50 epochs; negative log-likelihood loss\nRetain predictions at minimum validation loss",
        ),
        (
            "Score held-out links",
            "P(link) from class-1 probability\nPositive prediction when score > 0.5\nROC-AUC, PR-AUC, F1\nPrecision, Recall, MCC, TSS",
        ),
    ]

    row_specs = (
        (row1_y, row1, BLUE, WHITE, 1, xs),
        (row2_y, row2, TEAL, WHITE, 5, list(reversed(xs))),
        (row3_y, row3, ORANGE, WHITE, 9, xs),
    )
    for row_y, row, accent, fill, start_number, row_xs in row_specs:
        for idx, (title, body) in enumerate(row):
            rounded_box(
                ax,
                row_xs[idx],
                row_y,
                w,
                h,
                title,
                body,
                accent,
                fill,
                number=start_number + idx,
                title_size=10.1,
                body_size=7.8,
                line_spacing=1.22,
            )

    for row_y in (row1_y, row3_y):
        for idx in range(3):
            arrow(
                ax,
                (xs[idx] + w + 0.04, row_y + h / 2),
                (xs[idx + 1] - 0.04, row_y + h / 2),
            )

    # The middle lane runs right-to-left, producing a readable snake flow.
    for idx in range(3, 0, -1):
        arrow(
            ax,
            (xs[idx] - 0.04, row2_y + h / 2),
            (xs[idx - 1] + w + 0.04, row2_y + h / 2),
        )

    # Vertical connectors between processing lanes.
    arrow(
        ax,
        (xs[3] + w / 2, row1_y - 0.04),
        (xs[3] + w / 2, row2_y + h + 0.04),
        label="for every sampled pair",
        label_offset=(1.18, 0),
    )
    arrow(
        ax,
        (xs[0] + w / 2, row2_y - 0.04),
        (xs[0] + w / 2, row3_y + h + 0.04),
        label="train on graph examples",
        label_offset=(1.15, 0),
    )

    # Bottom reproducibility note.
    note_y = 1.36
    ax.add_patch(
        FancyBboxPatch(
            (1.20, note_y),
            16.95,
            0.86,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=PURPLE_LIGHT,
            edgecolor=PURPLE,
            linewidth=0.9,
        )
    )
    ax.text(1.48, note_y + 0.57, "REPRODUCIBILITY NOTE", fontsize=8.8, fontweight="bold", color=PURPLE)
    ax.text(
        4.05,
        note_y + 0.57,
        "The CUDA layer changes execution hardware only; sampling, splits, subgraphs, labels, model and metrics retain SEAL_original behavior.",
        fontsize=9,
        color=GRAY_700,
        va="center",
    )
    ax.text(
        1.48,
        note_y + 0.25,
        "Current campaign: 290 food webs × 10 runs · experiment seeds 12345–12354 reused across food webs · no Tukey filtering.",
        fontsize=8.8,
        color=GRAY_700,
        va="center",
    )

    ax.text(
        18.15,
        0.67,
        "Sources: Main.py · util_functions.py · pytorch_DGCNN · run_foodweb_seal.py",
        ha="right",
        va="center",
        fontsize=8.3,
        color=GRAY_500,
    )
    save_all(fig, "seal_original_flow_detailed")


if __name__ == "__main__":
    build_high_level()
    build_detailed()
