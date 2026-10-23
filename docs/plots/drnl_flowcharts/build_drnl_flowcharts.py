"""Generate high-level and worked-example DRNL diagrams.

The formula and edge-case behavior reproduce node_label() in
src/python/util_functions.py. Outputs are editable SVG plus PDF and PNG.
"""

from collections import deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


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
RED = "#B54A4A"
GRAY_900 = "#23313D"
GRAY_700 = "#4F5E69"
GRAY_500 = "#7B8993"
GRAY_300 = "#CBD5DC"
GRAY_100 = "#F4F7F8"
WHITE = "#FFFFFF"

LABEL_COLORS = {
    0: "#B8C1C7",
    1: PURPLE,
    2: TEAL,
    4: BLUE,
    5: ORANGE,
    10: RED,
}


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


def canvas(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def title_block(ax, title, subtitle, y):
    ax.text(0.65, y, title, ha="left", va="top", fontsize=23, fontweight="bold", color=NAVY)
    ax.text(0.65, y - 0.50, subtitle, ha="left", va="top", fontsize=11.5, color=GRAY_700)
    ax.plot([0.65, 1.85], [y - 0.84, y - 0.84], color=ORANGE, linewidth=4, solid_capstyle="round")


def box(ax, x, y, w, h, number, title, body, accent, fill, title_size=10.3, body_size=8.6):
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
    ax.add_patch(Rectangle((x, y + h - 0.12), w, 0.12, linewidth=0, facecolor=accent, zorder=3))
    ax.text(
        x + 0.25,
        y + h - 0.43,
        str(number),
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color=WHITE,
        bbox=dict(boxstyle="circle,pad=0.30", facecolor=accent, edgecolor="none"),
        zorder=4,
    )
    ax.text(
        x + 0.58,
        y + h - 0.43,
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=GRAY_900,
        zorder=4,
    )
    ax.text(
        x + 0.22,
        y + h - 0.83,
        body,
        ha="left",
        va="top",
        fontsize=body_size,
        color=GRAY_700,
        linespacing=1.30,
        zorder=4,
    )


def arrow(ax, start, end, color=GRAY_500, label=None, label_offset=(0, 0)):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.6,
            color=color,
            shrinkA=2,
            shrinkB=2,
            zorder=6,
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
            bbox=dict(boxstyle="round,pad=0.16", facecolor=WHITE, edgecolor="none"),
            zorder=7,
        )


def node(ax, x, y, text, color, radius=0.27, label=None, edgecolor=WHITE, text_size=9):
    ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=edgecolor, linewidth=1.8, zorder=5))
    ax.text(x, y, text, ha="center", va="center", fontsize=text_size, fontweight="bold", color=WHITE, zorder=6)
    if label is not None:
        ax.text(
            x + radius * 0.88,
            y + radius * 0.86,
            str(label),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color=WHITE,
            bbox=dict(boxstyle="circle,pad=0.20", facecolor=NAVY, edgecolor=WHITE, linewidth=0.8),
            zorder=7,
        )


def edge(ax, p1, p2, color=GRAY_500, linewidth=1.6, linestyle="-"):
    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        solid_capstyle="round",
        zorder=3,
    )


def save_all(fig, stem):
    for suffix, kwargs in (("svg", {}), ("pdf", {}), ("png", {"dpi": 240})):
        path = OUT_DIR / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=WHITE, **kwargs)
        print(f"[OK] {path}")
    plt.close(fig)


def shortest_distance(graph, source, destination, removed):
    """Unweighted distance while excluding one target node."""
    if source == removed or destination == removed:
        return None
    queue = deque([(source, 0)])
    visited = {source, removed}
    while queue:
        current, distance = queue.popleft()
        if current == destination:
            return distance
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    return None


def drnl_label(distance_u, distance_v):
    """Exact arithmetic used by util_functions.node_label()."""
    if distance_u is None or distance_v is None:
        return 0, None, None
    distance_sum = distance_u + distance_v
    quotient, remainder = divmod(distance_sum, 2)
    label = 1 + min(distance_u, distance_v) + quotient * (quotient + remainder - 1)
    return label, quotient, remainder


EXAMPLE_EDGES = [
    ("u", "a"),
    ("u", "c"),
    ("u", "z"),
    ("v", "b"),
    ("v", "c"),
    ("a", "d"),
    ("b", "d"),
    ("d", "e"),
    ("c", "f"),
    ("f", "g"),
]


def example_results():
    graph = {name: set() for name in ("u", "v", "a", "b", "c", "d", "e", "f", "g", "z")}
    for left, right in EXAMPLE_EDGES:
        graph[left].add(right)
        graph[right].add(left)

    rows = []
    for name in ("a", "b", "c", "d", "e", "f", "g", "z"):
        distance_u = shortest_distance(graph, name, "u", removed="v")
        distance_v = shortest_distance(graph, name, "v", removed="u")
        label, quotient, remainder = drnl_label(distance_u, distance_v)
        rows.append((name, distance_u, distance_v, quotient, remainder, label))

    expected = {
        "a": (1, 3, 2, 0, 4),
        "b": (3, 1, 2, 0, 4),
        "c": (1, 1, 1, 0, 2),
        "d": (2, 2, 2, 0, 5),
        "e": (3, 3, 3, 0, 10),
        "f": (2, 2, 2, 0, 5),
        "g": (3, 3, 3, 0, 10),
        "z": (1, None, None, None, 0),
    }
    assert {row[0]: row[1:] for row in rows} == expected
    return graph, rows


def build_high_level():
    fig, ax = canvas(18.4, 9.2, (0, 18.4), (0, 9.2))
    title_block(
        ax,
        "DRNL | High-level workflow",
        "Double-Radius Node Labeling converts local topology around a candidate link into structural node features",
        y=8.95,
    )

    stages = [
        (
            "Candidate link",
            "Select target nodes u and v\ninside an enclosing subgraph.\nThe target edge is not used\nfor classification.",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Two distance views",
            "For every node i:\n• distance to u with v removed\n• distance to v with u removed",
            BLUE,
            BLUE_LIGHT,
        ),
        (
            "Distance pair",
            "Represent the node by\n(dᵤ(i), dᵥ(i)).\nThis describes its position\nrelative to both targets.",
            TEAL,
            TEAL_LIGHT,
        ),
        (
            "DRNL mapping",
            "Let d = dᵤ + dᵥ,\nq = ⌊d/2⌋ and r = d mod 2.\n\nz(i) = 1 + min(dᵤ,dᵥ)\n       + q(q + r − 1)",
            ORANGE,
            ORANGE_LIGHT,
        ),
        (
            "Structural feature",
            "Assign one integer label z(i),\nthen one-hot encode it.\nDGCNN receives these labels\nas node features.",
            PURPLE,
            PURPLE_LIGHT,
        ),
    ]

    w, h, gap = 3.05, 3.10, 0.45
    x0, y = 0.65, 3.95
    for idx, (title, body, accent, fill) in enumerate(stages, start=1):
        x = x0 + (idx - 1) * (w + gap)
        box(ax, x, y, w, h, idx, title, body, accent, fill, title_size=10.2, body_size=8.8)
        if idx < len(stages):
            arrow(ax, (x + w + 0.04, y + h / 2), (x + w + gap - 0.04, y + h / 2))

    strip_y = 1.50
    ax.add_patch(
        FancyBboxPatch(
            (0.65, strip_y),
            17.05,
            1.32,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=GRAY_100,
            edgecolor=GRAY_300,
            linewidth=1,
        )
    )
    ax.text(0.93, strip_y + 0.96, "KEY PROPERTIES", fontsize=9.2, fontweight="bold", color=NAVY)
    properties = [
        ("Targets", "z(u) = z(v) = 1"),
        ("Unreachable", "z(i) = 0"),
        ("Input", "local topology only"),
        ("Output", "permutation-invariant structural roles"),
    ]
    starts = [0.93, 4.30, 8.20, 11.80]
    value_offsets = [0.92, 1.28, 0.92, 1.02]
    for (heading, value), x, value_offset in zip(properties, starts, value_offsets):
        ax.text(x, strip_y + 0.47, heading, fontsize=8.7, fontweight="bold", color=GRAY_900, va="center")
        ax.text(x + value_offset, strip_y + 0.47, value, fontsize=8.7, color=GRAY_700, va="center")

    ax.text(
        17.70,
        0.69,
        "Exact mapping used by SEAL_original · util_functions.node_label()",
        ha="right",
        va="center",
        fontsize=8.5,
        color=GRAY_500,
    )
    save_all(fig, "drnl_flow_high_level")


def draw_example_graph(ax, origin_x, origin_y, scale, labels):
    positions = {
        "u": (1.65, 4.65),
        "v": (5.55, 4.65),
        "a": (1.55, 3.20),
        "b": (5.65, 3.20),
        "c": (3.60, 3.75),
        "d": (3.60, 2.15),
        "e": (3.60, 0.78),
        "f": (4.75, 2.45),
        "g": (5.50, 1.25),
        "z": (0.45, 4.25),
    }
    positions = {name: (origin_x + x * scale, origin_y + y * scale) for name, (x, y) in positions.items()}

    for left, right in EXAMPLE_EDGES:
        edge(ax, positions[left], positions[right], color="#8A99A4", linewidth=1.8)
    edge(ax, positions["u"], positions["v"], color=PURPLE, linewidth=1.8, linestyle="--")
    ax.text(
        (positions["u"][0] + positions["v"][0]) / 2,
        positions["u"][1] + 0.22,
        "candidate link removed",
        ha="center",
        va="bottom",
        fontsize=7.8,
        color=PURPLE,
    )

    for name, position in positions.items():
        label = 1 if name in ("u", "v") else labels[name]
        node(ax, *position, name, LABEL_COLORS[label], radius=0.25 * scale, label=label, text_size=8.5)

    return positions


def build_detailed():
    _, rows = example_results()
    labels = {name: label for name, _, _, _, _, label in rows}

    fig, ax = canvas(18.8, 12.3, (0, 18.8), (0, 12.3))
    title_block(
        ax,
        "DRNL | Detailed worked example",
        "Distances are measured independently to each target; the pair is compressed into one discrete structural label",
        y=12.05,
    )

    # Left: example graph.
    left_x, panel_y, left_w, panel_h = 0.65, 2.65, 7.55, 7.95
    ax.add_patch(
        FancyBboxPatch(
            (left_x, panel_y),
            left_w,
            panel_h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            facecolor=GRAY_100,
            edgecolor=GRAY_300,
            linewidth=1.1,
        )
    )
    ax.text(left_x + 0.30, panel_y + panel_h - 0.42, "A. Enclosing-subgraph example", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(
        left_x + 0.30,
        panel_y + panel_h - 0.78,
        "Badge = computed DRNL label z(i). The dashed target edge is excluded from the classifier input.",
        fontsize=8.5,
        color=GRAY_700,
    )
    draw_example_graph(ax, origin_x=left_x + 0.38, origin_y=panel_y + 0.76, scale=0.95, labels=labels)

    legend_y = panel_y + 0.47
    ax.text(left_x + 0.30, legend_y, "Label colours", fontsize=8.3, fontweight="bold", color=GRAY_900, va="center")
    legend_x = left_x + 1.55
    for label in (0, 1, 2, 4, 5, 10):
        node(ax, legend_x, legend_y, str(label), LABEL_COLORS[label], radius=0.17, text_size=6.8)
        legend_x += 0.67

    # Right top: exact distance and labeling rules.
    right_x, right_w = 8.55, 9.60
    formula_y, formula_h = 7.55, 3.05
    ax.add_patch(
        FancyBboxPatch(
            (right_x, formula_y),
            right_w,
            formula_h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            facecolor=ORANGE_LIGHT,
            edgecolor=ORANGE,
            linewidth=1.1,
        )
    )
    ax.text(right_x + 0.30, formula_y + formula_h - 0.42, "B. Exact mapping", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(
        right_x + 0.35,
        formula_y + 1.99,
        "dᵤ(i): shortest path from i to u after removing v",
        fontsize=9.1,
        color=GRAY_700,
    )
    ax.text(
        right_x + 0.35,
        formula_y + 1.64,
        "dᵥ(i): shortest path from i to v after removing u",
        fontsize=9.1,
        color=GRAY_700,
    )
    ax.text(
        right_x + 0.35,
        formula_y + 0.92,
        r"$d=d_u+d_v,\quad q=\lfloor d/2\rfloor,\quad r=d\ \mathrm{mod}\ 2$",
        fontsize=13,
        color=GRAY_900,
    )
    ax.text(
        right_x + 0.35,
        formula_y + 0.36,
        r"$z(i)=1+\min(d_u,d_v)+q\,(q+r-1)$",
        fontsize=15,
        fontweight="bold",
        color=ORANGE,
    )
    ax.text(
        right_x + 6.80,
        formula_y + 1.80,
        "Special cases",
        fontsize=9,
        fontweight="bold",
        color=PURPLE,
    )
    ax.text(
        right_x + 6.80,
        formula_y + 1.42,
        "z(u) = z(v) = 1\n∞ distance → z(i) = 0",
        fontsize=9.2,
        color=GRAY_700,
        linespacing=1.35,
    )

    # Right bottom: table calculated from the example graph.
    table_y, table_h = 2.65, 4.55
    ax.add_patch(
        FancyBboxPatch(
            (right_x, table_y),
            right_w,
            table_h,
            boxstyle="round,pad=0.02,rounding_size=0.14",
            facecolor=WHITE,
            edgecolor=TEAL,
            linewidth=1.1,
            zorder=0,
        )
    )
    ax.text(right_x + 0.30, table_y + table_h - 0.40, "C. Node-by-node calculation", fontsize=12, fontweight="bold", color=NAVY)

    columns = ["Node i", "dᵤ(i)", "dᵥ(i)", "q", "r", "z(i)"]
    col_x = [right_x + 0.55, right_x + 2.15, right_x + 3.55, right_x + 4.90, right_x + 6.05, right_x + 7.48]
    header_y = table_y + table_h - 0.92
    for heading, x in zip(columns, col_x):
        ax.text(x, header_y, heading, ha="center", va="center", fontsize=8.8, fontweight="bold", color=WHITE)
    ax.add_patch(
        FancyBboxPatch(
            (right_x + 0.25, header_y - 0.23),
            right_w - 0.50,
            0.46,
            boxstyle="round,pad=0.01,rounding_size=0.06",
            facecolor=TEAL,
            edgecolor="none",
            zorder=1,
        )
    )

    row_start_y, row_step = header_y - 0.50, 0.39
    for idx, (name, distance_u, distance_v, quotient, remainder, label) in enumerate(rows):
        y = row_start_y - idx * row_step
        if idx % 2 == 0:
            ax.add_patch(Rectangle((right_x + 0.25, y - 0.18), right_w - 0.50, 0.36, facecolor=TEAL_LIGHT, edgecolor="none", zorder=1))
        values = [
            name,
            "∞" if distance_u is None else str(distance_u),
            "∞" if distance_v is None else str(distance_v),
            "—" if quotient is None else str(quotient),
            "—" if remainder is None else str(remainder),
            str(label),
        ]
        for value, x in zip(values, col_x):
            kwargs = dict(fontsize=8.7, color=GRAY_700)
            if x == col_x[-1]:
                kwargs.update(fontweight="bold", color=LABEL_COLORS[label])
            ax.text(x, y, value, ha="center", va="center", **kwargs)

    # Bottom encoding strip.
    strip_y = 1.24
    ax.add_patch(
        FancyBboxPatch(
            (0.65, strip_y),
            17.50,
            0.92,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=PURPLE_LIGHT,
            edgecolor=PURPLE,
            linewidth=1,
        )
    )
    ax.text(0.93, strip_y + 0.58, "MODEL INPUT", fontsize=9, fontweight="bold", color=PURPLE, va="center")
    ax.text(2.25, strip_y + 0.58, "integer label z(i)", fontsize=9.1, color=GRAY_700, va="center")
    arrow(ax, (4.25, strip_y + 0.58), (5.10, strip_y + 0.58), color=PURPLE)
    ax.text(5.35, strip_y + 0.58, "one-hot DRNL node tag", fontsize=9.1, color=GRAY_700, va="center")
    arrow(ax, (7.65, strip_y + 0.58), (8.50, strip_y + 0.58), color=PURPLE)
    ax.text(8.75, strip_y + 0.58, "DGCNN graph convolutions", fontsize=9.1, color=GRAY_700, va="center")
    ax.text(
        17.85,
        strip_y + 0.58,
        "Topology becomes a learnable categorical feature",
        fontsize=8.7,
        color=GRAY_500,
        ha="right",
        va="center",
    )

    ax.text(
        18.15,
        0.55,
        "Formula and edge cases validated against src/python/util_functions.py",
        ha="right",
        va="center",
        fontsize=8.4,
        color=GRAY_500,
    )
    save_all(fig, "drnl_flow_detailed")


if __name__ == "__main__":
    build_high_level()
    build_detailed()
