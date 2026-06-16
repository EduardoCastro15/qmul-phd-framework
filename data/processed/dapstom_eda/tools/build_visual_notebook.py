import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path


EDA_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = EDA_DIR / "tables"
FIG_DIR = EDA_DIR / "figures"
NOTEBOOK_PATH = EDA_DIR / "DAPSTOM_EDA_visuals.ipynb"

PALETTE = ["#36648B", "#4C8C5A", "#C46A2D", "#7E5AA2", "#A23E48", "#548C8C"]
PRED_COLOR = "#36648B"
PREY_COLOR = "#4C8C5A"
NON_TROPHIC_COLOR = "#C46A2D"
GRID_COLOR = "#D8DEE4"
TEXT_COLOR = "#1F2933"
MUTED_TEXT = "#52616B"
BG = "#FFFFFF"


def read_rows(name):
    with (TABLE_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def as_int(value):
    return int(round(as_float(value)))


def fmt_int(value):
    return f"{as_int(value):,}"


def pct(numer, denom):
    denom = as_float(denom)
    if denom == 0:
        return "n/a"
    return f"{100 * as_float(numer) / denom:.1f}%"


def esc(value):
    return html.escape("" if value is None else str(value), quote=True)


def truncate(value, max_len):
    value = "" if value is None else str(value)
    return value if len(value) <= max_len else value[: max_len - 1] + "..."


def svg_doc(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f'<rect width="{width}" height="{height}" fill="{BG}"/>\n'
        f'<style>text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
        f'fill:{TEXT_COLOR};font-size:13px}} .title{{font-size:22px;font-weight:700}} '
        f'.subtitle{{font-size:13px;fill:{MUTED_TEXT}}} .small{{font-size:11px;fill:{MUTED_TEXT}}}</style>\n'
        f"{body}\n</svg>\n"
    )


def write_svg(name, width, height, body):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    path.write_text(svg_doc(width, height, body), encoding="utf-8")
    return path


def horizontal_bar_chart(name, title, subtitle, rows, value_key, label_key, limit=12, color="#36648B"):
    rows = rows[:limit]
    width = 980
    row_h = 34
    top = 86
    left = 270
    right = 80
    height = top + row_h * len(rows) + 52
    max_v = max(as_float(r[value_key]) for r in rows) if rows else 1
    plot_w = width - left - right
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for i, r in enumerate(rows):
        y = top + i * row_h
        v = as_float(r[value_key])
        bar_w = 0 if max_v == 0 else plot_w * v / max_v
        body.append(f'<text x="{left - 14}" y="{y + 18}" text-anchor="end">{esc(truncate(r[label_key], 34))}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="20" rx="3" fill="#EEF2F5"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="20" rx="3" fill="{color}"/>')
        body.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 15}" class="small">{fmt_int(v)}</text>')
    write_svg(name, width, height, "\n".join(body))


def vertical_bar_chart(name, title, subtitle, rows, value_key, label_key, color="#4C8C5A"):
    width, height = 1040, 520
    left, right, top, bottom = 70, 40, 78, 88
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max(as_float(r[value_key]) for r in rows) if rows else 1
    bar_gap = 5
    bar_w = max(8, (plot_w - bar_gap * (len(rows) - 1)) / max(1, len(rows)))
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="{GRID_COLOR}"/>',
    ]
    for frac in [0.25, 0.5, 0.75, 1.0]:
        y = top + plot_h * (1 - frac)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-dasharray="2 4"/>')
        body.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" class="small">{fmt_int(max_v * frac)}</text>')
    for i, r in enumerate(rows):
        v = as_float(r[value_key])
        h = 0 if max_v == 0 else plot_h * v / max_v
        x = left + i * (bar_w + bar_gap)
        y = top + plot_h - h
        body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
        label = str(as_int(r[label_key]))
        rotate = -45 if len(rows) > 14 else 0
        if rotate:
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 22}" transform="rotate(-45 {x + bar_w / 2:.1f},{top + plot_h + 22})" text-anchor="end" class="small">{esc(label)}</text>')
        else:
            body.append(f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 20}" text-anchor="middle" class="small">{esc(label)}</text>')
    write_svg(name, width, height, "\n".join(body))


def spatial_coverage_chart():
    rows = read_rows("haul_spatial_resolution")
    width, height = 900, 360
    left, top = 210, 92
    plot_w = 560
    row_h = 42
    body = [
        '<text x="30" y="34" class="title">Spatial resolution available by haul</text>',
        '<text x="30" y="58" class="subtitle">Point coordinates are partial; ICES division and sea are complete.</text>',
    ]
    for i, r in enumerate(rows):
        label = r["resolution"].strip()
        coverage = 100 * as_float(r["haul_rows"]) / as_float(r["total_hauls"])
        y = top + i * row_h
        body.append(f'<text x="{left - 14}" y="{y + 20}" text-anchor="end">{esc(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="24" rx="4" fill="#EEF2F5"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_w * coverage / 100:.1f}" height="24" rx="4" fill="{PALETTE[i % len(PALETTE)]}"/>')
        body.append(f'<text x="{left + plot_w + 12}" y="{y + 18}" class="small">{coverage:.1f}%</text>')
    write_svg("spatial_resolution_coverage.svg", width, height, "\n".join(body))


def pooled_comparison_chart():
    rows = read_rows("predator_pooling_summary")
    total_rows = sum(as_float(r["predator_rows"]) for r in rows)
    total_stomachs = sum(as_float(r["total_stomachs"]) for r in rows)
    width, height = 920, 300
    left, top = 210, 102
    bar_w = 580
    bar_h = 34
    colors = {"n": "#36648B", "y": "#C46A2D", "": "#8B949E"}
    labels = {"n": "individual rows", "y": "pooled rows", "": "missing"}
    body = [
        '<text x="30" y="34" class="title">Pooled records change the sampling unit</text>',
        '<text x="30" y="58" class="subtitle">Pooled rows are few, but represent many stomachs.</text>',
    ]
    for j, (metric, total, value_key) in enumerate([
        ("Predator rows", total_rows, "predator_rows"),
        ("Represented stomachs", total_stomachs, "total_stomachs"),
    ]):
        y = top + j * 78
        x = left
        body.append(f'<text x="{left - 18}" y="{y + 23}" text-anchor="end">{metric}</text>')
        for r in rows:
            key = r["pooled"]
            v = as_float(r[value_key])
            w = 0 if total == 0 else bar_w * v / total
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_h}" fill="{colors.get(key, "#8B949E")}"/>')
            if w > 58:
                body.append(f'<text x="{x + w / 2:.1f}" y="{y + 22}" text-anchor="middle" fill="#FFFFFF" class="small">{pct(v, total)}</text>')
            x += w
    legend_x = left
    legend_y = top + 170
    for i, key in enumerate(["n", "y", ""]):
        x = legend_x + i * 170
        body.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{colors[key]}"/>')
        body.append(f'<text x="{x + 20}" y="{legend_y + 12}" class="small">{esc(labels[key])}</text>')
    write_svg("pooled_rows_vs_stomachs.svg", width, height, "\n".join(body))


def missingness_chart():
    rows = []
    for r in read_rows("critical_field_missingness"):
        missing = as_int(r["null_rows"]) + as_int(r["blank_text_rows"])
        if missing:
            rows.append({
                "label": f"{r['field_name']} ({r['table_name'].split()[0]})",
                "missing": missing,
                "total": as_int(r["total_rows"]),
            })
    rows.sort(key=lambda r: r["missing"] / r["total"], reverse=True)
    chart_rows = [{"label": r["label"], "value": r["missing"]} for r in rows]
    horizontal_bar_chart(
        "critical_missingness.svg",
        "Critical-field missingness",
        "Null and blank counts in fields needed for spatial, temporal, and evidence handling.",
        chart_rows,
        "value",
        "label",
        limit=10,
        color="#A23E48",
    )


def heatmap_sea_decade():
    raw = read_rows("edge_pairs_by_sea_decade")
    totals_by_sea = defaultdict(float)
    values = {}
    decades = set()
    for r in raw:
        sea = r.get("sea") or r.get("SEA")
        decade = str(as_int(r.get("decade") or r.get("DECADE")))
        value = as_float(r["unique_predator_prey_pairs"])
        totals_by_sea[sea] += value
        values[(sea, decade)] = value
        decades.add(decade)
    seas = [s for s, _ in sorted(totals_by_sea.items(), key=lambda kv: kv[1], reverse=True)[:10]]
    decades = sorted(decades, key=lambda x: int(x))
    max_v = max(values.values()) if values else 1
    cell_w, cell_h = 42, 30
    left, top = 150, 86
    width = left + cell_w * len(decades) + 50
    height = top + cell_h * len(seas) + 88
    body = [
        '<text x="30" y="34" class="title">Network density by sea and decade</text>',
        '<text x="30" y="58" class="subtitle">Unique predator-prey name pairs in the strongest sea-decade strata.</text>',
    ]
    for j, decade in enumerate(decades):
        x = left + j * cell_w + cell_w / 2
        body.append(f'<text x="{x:.1f}" y="{top - 12}" text-anchor="middle" class="small">{esc(decade)}</text>')
    for i, sea in enumerate(seas):
        y = top + i * cell_h
        body.append(f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" class="small">{esc(truncate(sea, 18))}</text>')
        for j, decade in enumerate(decades):
            x = left + j * cell_w
            value = values.get((sea, decade), 0.0)
            intensity = 0 if max_v == 0 else value / max_v
            fill = interpolate_color("#F3F6F8", "#36648B", intensity)
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{fill}" stroke="#FFFFFF"/>')
            if value >= max_v * 0.38:
                body.append(f'<text x="{x + cell_w / 2:.1f}" y="{y + 19}" text-anchor="middle" fill="#FFFFFF" class="small">{fmt_int(value)}</text>')
    body.append(f'<text x="{left}" y="{height - 24}" class="small">Darker cells indicate more unique predator-prey pairs.</text>')
    write_svg("sea_decade_edge_heatmap.svg", width, height, "\n".join(body))


def interpolate_color(low, high, t):
    t = max(0.0, min(1.0, t))
    def rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    a = rgb(low)
    b = rgb(high)
    c = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#" + "".join(f"{v:02X}" for v in c)


def edge_network(name, title, subtitle, rows, limit=36, include_negative=False):
    rows = rows[:limit]
    pred_weight = defaultdict(float)
    prey_weight = defaultdict(float)
    edges = []
    for r in rows:
        pred = r["predator_name"]
        prey = r["prey_name"]
        w = as_float(r["prey_records"])
        prey_tsn = as_float(r.get("prey_tsn", 0))
        pred_weight[pred] += w
        prey_weight[prey] += w
        edges.append((pred, prey, w, prey_tsn))
    predators = [k for k, _ in sorted(pred_weight.items(), key=lambda kv: kv[1], reverse=True)]
    preys = [k for k, _ in sorted(prey_weight.items(), key=lambda kv: kv[1], reverse=True)]
    max_nodes = max(len(predators), len(preys), 1)
    height = max(660, 104 + max_nodes * 25)
    width = 1160
    x_pred, x_prey = 210, 910
    top, bottom = 92, 42
    y_span = height - top - bottom
    def y_positions(labels):
        if len(labels) == 1:
            return {labels[0]: top + y_span / 2}
        return {label: top + i * y_span / (len(labels) - 1) for i, label in enumerate(labels)}
    pred_y = y_positions(predators)
    prey_y = y_positions(preys)
    max_w = max((e[2] for e in edges), default=1)
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
        f'<text x="{x_pred}" y="{top - 24}" text-anchor="middle" class="small">Predators</text>',
        f'<text x="{x_prey}" y="{top - 24}" text-anchor="middle" class="small">Prey</text>',
    ]
    for pred, prey, w, prey_tsn in edges:
        y1 = pred_y[pred]
        y2 = prey_y[prey]
        sw = 0.7 + 5.5 * math.log1p(w) / math.log1p(max_w)
        edge_color = NON_TROPHIC_COLOR if include_negative and prey_tsn < 0 else "#8795A1"
        body.append(
            f'<path d="M{x_pred + 10},{y1:.1f} C{x_pred + 260},{y1:.1f} {x_prey - 260},{y2:.1f} {x_prey - 10},{y2:.1f}" '
            f'fill="none" stroke="{edge_color}" stroke-opacity="0.34" stroke-width="{sw:.2f}"/>'
        )
    max_pred = max(pred_weight.values(), default=1)
    max_prey = max(prey_weight.values(), default=1)
    for pred in predators:
        y = pred_y[pred]
        r = 5 + 9 * math.sqrt(pred_weight[pred] / max_pred)
        body.append(f'<circle cx="{x_pred}" cy="{y:.1f}" r="{r:.1f}" fill="{PRED_COLOR}" stroke="#FFFFFF" stroke-width="1.5"/>')
        body.append(f'<text x="{x_pred - 18}" y="{y + 4:.1f}" text-anchor="end" class="small">{esc(truncate(pred, 24))}</text>')
    for prey in preys:
        y = prey_y[prey]
        r = 4 + 8 * math.sqrt(prey_weight[prey] / max_prey)
        is_non_trophic = any(e[1] == prey and e[3] < 0 for e in edges)
        color = NON_TROPHIC_COLOR if is_non_trophic else PREY_COLOR
        body.append(f'<circle cx="{x_prey}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="#FFFFFF" stroke-width="1.5"/>')
        body.append(f'<text x="{x_prey + 18}" y="{y + 4:.1f}" class="small">{esc(truncate(prey, 36))}</text>')
    legend_y = height - 24
    body.append(f'<circle cx="30" cy="{legend_y - 4}" r="6" fill="{PRED_COLOR}"/><text x="44" y="{legend_y}" class="small">predator</text>')
    body.append(f'<circle cx="130" cy="{legend_y - 4}" r="6" fill="{PREY_COLOR}"/><text x="144" y="{legend_y}" class="small">prey</text>')
    if include_negative:
        body.append(f'<circle cx="220" cy="{legend_y - 4}" r="6" fill="{NON_TROPHIC_COLOR}"/><text x="234" y="{legend_y}" class="small">negative TSN/non-trophic category</text>')
    write_svg(name, width, height, "\n".join(body))


def build_figures():
    horizontal_bar_chart(
        "table_row_counts.svg",
        "DAPSTOM table sizes",
        "Rows in user tables exposed by the Access database.",
        read_rows("table_row_counts"),
        "row_count",
        "table_name",
        limit=10,
        color="#36648B",
    )
    vertical_bar_chart(
        "hauls_by_decade.svg",
        "Haul records by decade",
        "Temporal spread in the HAUL table; early historical data are sparse.",
        read_rows("haul_temporal_by_decade"),
        "haul_rows",
        "decade",
        color="#4C8C5A",
    )
    horizontal_bar_chart(
        "hauls_by_sea.svg",
        "Haul records by sea",
        "Top sea-level groups by haul count.",
        read_rows("haul_spatial_by_sea"),
        "haul_rows",
        "sea",
        limit=12,
        color="#7E5AA2",
    )
    spatial_coverage_chart()
    pooled_comparison_chart()
    missingness_chart()
    horizontal_bar_chart(
        "top_predators.svg",
        "Most common predator records",
        "Top predators by row count in the PREDATOR table.",
        read_rows("top_predators"),
        "predator_rows",
        "predator_name",
        limit=12,
        color="#36648B",
    )
    horizontal_bar_chart(
        "negative_tsn_categories.svg",
        "Negative TSN prey categories",
        "Empty, unknown, digested, and other non-standard categories to model separately.",
        read_rows("negative_tsn_prey_categories"),
        "prey_rows",
        "prey_name",
        limit=12,
        color="#C46A2D",
    )
    heatmap_sea_decade()
    edge_network(
        "network_top_raw_edges.svg",
        "Raw top predator-prey rows",
        "Includes empty/non-standard prey categories; useful as a cautionary view.",
        read_rows("top_predator_prey_edges"),
        limit=30,
        include_negative=True,
    )
    edge_network(
        "network_positive_prey_tsn_edges.svg",
        "Top predator-prey network after filtering negative prey TSN",
        "A cleaner first-pass trophic network based on the most frequent positive-TSN prey rows.",
        read_rows("top_predator_prey_edges_positive_prey_tsn"),
        limit=42,
        include_negative=False,
    )


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def figure_md(path, caption):
    return f'<img src="{path}" alt="{esc(caption)}" style="max-width:100%; height:auto;">\n\n*{caption}*'


def build_notebook():
    cells = [
        markdown_cell(
            "# DAPSTOM 6.4 Visual EDA\n\n"
            "This notebook complements `DAPSTOM_EDA.ipynb` with visual summaries and first-pass network views. "
            "Figures are generated as SVG files from the derived CSV summaries under `tables/`."
        ),
        markdown_cell(
            "## How to regenerate\n\n"
            "From the repo root:\n\n"
            "```bash\n"
            "bash data/processed/dapstom_eda/tools/run_extractor.sh\n"
            "python3 data/processed/dapstom_eda/tools/build_visual_notebook.py\n"
            "```\n\n"
            "The plots use derived summaries only. The source Access database is not modified."
        ),
        code_cell(
            "from pathlib import Path\n"
            "FIG_DIR = Path('figures')\n"
            "if not FIG_DIR.exists():\n"
            "    FIG_DIR = Path('data/processed/dapstom_eda/figures')\n"
            "sorted(p.name for p in FIG_DIR.glob('*.svg'))"
        ),
        markdown_cell("## 1. Database Scale\n\n" + figure_md("figures/table_row_counts.svg", "Main table sizes in the Access database.")),
        markdown_cell("## 2. Temporal Coverage\n\n" + figure_md("figures/hauls_by_decade.svg", "Haul records aggregated by decade.")),
        markdown_cell("## 3. Spatial Coverage\n\n" + figure_md("figures/hauls_by_sea.svg", "Top sea-level groups by haul count.")),
        markdown_cell(figure_md("figures/spatial_resolution_coverage.svg", "Availability of point coordinates, ICES rectangle, ICES division, and sea.")),
        markdown_cell("## 4. Sampling Unit: Pooled vs Individual\n\n" + figure_md("figures/pooled_rows_vs_stomachs.svg", "Pooled rows are fewer but represent many stomachs.")),
        markdown_cell("## 5. Missingness\n\n" + figure_md("figures/critical_missingness.svg", "Missingness in fields needed for spatial, temporal, and evidence-aware modelling.")),
        markdown_cell("## 6. Dominant Taxa and Non-standard Categories\n\n" + figure_md("figures/top_predators.svg", "Most common predator records.")),
        markdown_cell(figure_md("figures/negative_tsn_categories.svg", "Negative prey TSN categories that should usually be separated from ordinary trophic edges.")),
        markdown_cell("## 7. Candidate Food-web Strata\n\n" + figure_md("figures/sea_decade_edge_heatmap.svg", "Sea-decade strata ranked by unique predator-prey pairs.")),
        markdown_cell(
            "## 8. Network Views\n\n"
            "The first network intentionally includes raw top rows, so empty/non-standard categories appear. "
            "The second network filters to positive prey TSN values and is closer to a first-pass trophic network."
        ),
        markdown_cell(figure_md("figures/network_top_raw_edges.svg", "Raw top edges, including non-trophic/negative-TSN categories.")),
        markdown_cell(figure_md("figures/network_positive_prey_tsn_edges.svg", "Top predator-prey edges after filtering negative prey TSN values.")),
        markdown_cell(
            "## Interpretation Notes\n\n"
            "- Use the filtered positive-TSN network for initial food-web discussions.\n"
            "- Keep the raw network as a quality-control figure: it shows why `Empty`, digested remains, and unknown categories must be handled separately.\n"
            "- For modelling, the next step is not to use only the top edges. Instead, build a full derived edge table with `min_num`, `cpw`, `pooled`, `num_stomachs`, spatial grain, decade, and provenance.\n"
            "- Sea-decade strata are a pragmatic first grouping for supervisor discussion; ICES rectangle/haul-level networks will need sample-size checks."
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build_figures()
    build_notebook()
    print(NOTEBOOK_PATH)
