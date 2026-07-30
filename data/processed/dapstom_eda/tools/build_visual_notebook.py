import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path

from derive_network_metrics import derive


EDA_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = EDA_DIR / "tables"
FIG_DIR = EDA_DIR / "figures"
NOTEBOOK_PATH = EDA_DIR / "DAPSTOM_EDA_visuals.ipynb"

PALETTE = ["#36648B", "#4C8C5A", "#C46A2D", "#7E5AA2", "#A23E48", "#548C8C"]
PRED_COLOR = "#36648B"
PREY_COLOR = "#4C8C5A"
NON_TROPHIC_COLOR = "#C46A2D"
BOTH_ROLE_COLOR = "#7E5AA2"
GRID_COLOR = "#D8DEE4"
TEXT_COLOR = "#1F2933"
MUTED_TEXT = "#52616B"
BG = "#FFFFFF"
FONT_FAMILY = '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif'
BASE_FONT_SIZE = 13
TITLE_FONT_SIZE = 22
SUBTITLE_FONT_SIZE = 13
SMALL_FONT_SIZE = 11


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


def fmt_num(value, digits=2):
    number = as_float(value)
    absolute = abs(number)
    if absolute >= 10_000:
        return f"{number:,.0f}"
    if absolute >= 100:
        return f"{number:,.1f}"
    if 0 < absolute < 0.01:
        return f"{number:.3g}"
    return f"{number:,.{digits}f}"


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
        f'<style>text{{font-family:{FONT_FAMILY};fill:{TEXT_COLOR};font-size:{BASE_FONT_SIZE}px}} '
        f'.title{{font-size:{TITLE_FONT_SIZE}px;font-weight:700}} '
        f'.subtitle{{font-size:{SUBTITLE_FONT_SIZE}px;fill:{MUTED_TEXT}}} '
        f'.small{{font-size:{SMALL_FONT_SIZE}px;fill:{MUTED_TEXT}}}</style>\n'
        f"{body}\n</svg>\n"
    )


def write_svg(name, width, height, body):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    path.write_text(svg_doc(width, height, body), encoding="utf-8")
    return path


def horizontal_bar_chart(
    name,
    title,
    subtitle,
    rows,
    value_key,
    label_key,
    limit=12,
    color="#36648B",
    width=980,
    row_height=34,
    left=270,
    right=80,
    top=86,
    bar_background="#EEF2F5",
):
    rows = rows[:limit]
    height = top + row_height * len(rows) + 52
    max_v = max(as_float(r[value_key]) for r in rows) if rows else 1
    plot_w = width - left - right
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for i, r in enumerate(rows):
        y = top + i * row_height
        v = as_float(r[value_key])
        bar_w = 0 if max_v == 0 else plot_w * v / max_v
        body.append(f'<text x="{left - 14}" y="{y + 18}" text-anchor="end">{esc(truncate(r[label_key], 34))}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_w}" height="20" rx="3" fill="{bar_background}"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="20" rx="3" fill="{color}"/>')
        body.append(f'<text x="{left + bar_w + 8:.1f}" y="{y + 15}" class="small">{fmt_int(v)}</text>')
    write_svg(name, width, height, "\n".join(body))


def vertical_bar_chart(
    name,
    title,
    subtitle,
    rows,
    value_key,
    label_key,
    color="#4C8C5A",
    width=1040,
    height=520,
    left=70,
    right=40,
    top=78,
    bottom=88,
    bar_gap=5,
):
    plot_w, plot_h = width - left - right, height - top - bottom
    max_v = max(as_float(r[value_key]) for r in rows) if rows else 1
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


def spatial_coverage_chart(
    name="spatial_resolution_coverage.svg",
    title="Spatial resolution available by haul",
    subtitle="Point coordinates are partial; ICES division and sea are complete.",
    width=900,
    height=360,
    left=210,
    top=92,
    plot_width=560,
    row_height=42,
    colors=None,
):
    rows = read_rows("haul_spatial_resolution")
    colors = colors or PALETTE
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for i, r in enumerate(rows):
        label = r["resolution"].strip()
        coverage = 100 * as_float(r["haul_rows"]) / as_float(r["total_hauls"])
        y = top + i * row_height
        body.append(f'<text x="{left - 14}" y="{y + 20}" text-anchor="end">{esc(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_width}" height="24" rx="4" fill="#EEF2F5"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{plot_width * coverage / 100:.1f}" height="24" rx="4" fill="{colors[i % len(colors)]}"/>')
        body.append(f'<text x="{left + plot_width + 12}" y="{y + 18}" class="small">{coverage:.1f}%</text>')
    write_svg(name, width, height, "\n".join(body))


def pooled_comparison_chart(
    name="pooled_rows_vs_stomachs.svg",
    title="Pooled records change the sampling unit",
    subtitle="Pooled rows are few, but represent many stomachs.",
    width=920,
    height=300,
    left=210,
    top=102,
    bar_width=580,
    bar_height=34,
    colors=None,
    labels=None,
):
    rows = read_rows("predator_pooling_summary")
    total_rows = sum(as_float(r["predator_rows"]) for r in rows)
    total_stomachs = sum(as_float(r["total_stomachs"]) for r in rows)
    colors = colors or {"n": PRED_COLOR, "y": NON_TROPHIC_COLOR, "": "#8B949E"}
    labels = labels or {"n": "individual rows", "y": "pooled rows", "": "missing"}
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
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
            w = 0 if total == 0 else bar_width * v / total
            body.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="{bar_height}" fill="{colors.get(key, "#8B949E")}"/>')
            if w > 58:
                body.append(f'<text x="{x + w / 2:.1f}" y="{y + 22}" text-anchor="middle" fill="#FFFFFF" class="small">{pct(v, total)}</text>')
            x += w
    legend_x = left
    legend_y = top + 170
    for i, key in enumerate(["n", "y", ""]):
        x = legend_x + i * 170
        body.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{colors[key]}"/>')
        body.append(f'<text x="{x + 20}" y="{legend_y + 12}" class="small">{esc(labels[key])}</text>')
    write_svg(name, width, height, "\n".join(body))


def missingness_chart(
    name="critical_missingness.svg",
    title="Critical-field missingness",
    subtitle="Null and blank counts in fields needed for spatial, temporal, and evidence handling.",
    limit=10,
    color="#A23E48",
):
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
        name,
        title,
        subtitle,
        chart_rows,
        "value",
        "label",
        limit=limit,
        color=color,
    )


def heatmap_sea_decade(
    name="sea_decade_edge_heatmap.svg",
    title="Resolved link richness by sea and decade",
    subtitle="All sea-decade strata; unique predator-prey name pairs after requiring prey TSN > 0.",
    sea_limit=10,
    cell_width=42,
    cell_height=30,
    low_color="#F3F6F8",
    high_color="#36648B",
    label_threshold=0.38,
):
    raw = read_rows("edge_pairs_by_sea_decade_positive_prey_tsn")
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
    seas = [
        sea
        for sea, _ in sorted(
            totals_by_sea.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:sea_limit]
    ]
    decades = sorted(decades, key=lambda x: int(x))
    max_v = max(values.values()) if values else 1
    left, top = 150, 86
    width = left + cell_width * len(decades) + 50
    height = top + cell_height * len(seas) + 88
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for j, decade in enumerate(decades):
        x = left + j * cell_width + cell_width / 2
        body.append(f'<text x="{x:.1f}" y="{top - 12}" text-anchor="middle" class="small">{esc(decade)}</text>')
    for i, sea in enumerate(seas):
        y = top + i * cell_height
        body.append(f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" class="small">{esc(truncate(sea, 18))}</text>')
        for j, decade in enumerate(decades):
            x = left + j * cell_width
            value = values.get((sea, decade), 0.0)
            intensity = 0 if max_v == 0 else value / max_v
            fill = interpolate_color(low_color, high_color, intensity)
            body.append(f'<rect x="{x}" y="{y}" width="{cell_width - 2}" height="{cell_height - 2}" fill="{fill}" stroke="#FFFFFF"/>')
            if value >= max_v * label_threshold:
                body.append(f'<text x="{x + cell_width / 2:.1f}" y="{y + 19}" text-anchor="middle" fill="#FFFFFF" class="small">{fmt_int(value)}</text>')
    body.append(f'<text x="{left}" y="{height - 24}" class="small">Darker cells indicate more unique predator-prey pairs.</text>')
    write_svg(name, width, height, "\n".join(body))


def interpolate_color(low, high, t):
    t = max(0.0, min(1.0, t))
    def rgb(hex_color):
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    a = rgb(low)
    b = rgb(high)
    c = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#" + "".join(f"{v:02X}" for v in c)


def edge_network(
    name,
    title,
    subtitle,
    rows,
    limit=36,
    include_negative=False,
    width=1160,
    minimum_height=660,
    row_spacing=25,
    predator_x=210,
    prey_x=910,
    edge_color="#8795A1",
    edge_opacity=0.34,
    negative_color=None,
):
    rows = rows[:limit]
    negative_color = negative_color or NON_TROPHIC_COLOR
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
    height = max(minimum_height, 104 + max_nodes * row_spacing)
    x_pred, x_prey = predator_x, prey_x
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
        '<defs><marker id="flow-arrow" markerUnits="userSpaceOnUse" markerWidth="7" markerHeight="7" '
        'refX="6" refY="3.5" orient="auto">'
        '<path d="M0,0 L0,7 L7,3.5 z" fill="#657786"/></marker></defs>',
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
        f'<text x="{x_pred}" y="{top - 24}" text-anchor="middle" class="small">Predators</text>',
        f'<text x="{x_prey}" y="{top - 24}" text-anchor="middle" class="small">Prey</text>',
    ]
    for pred, prey, w, prey_tsn in edges:
        y1 = pred_y[pred]
        y2 = prey_y[prey]
        sw = 0.7 + 5.5 * math.log1p(w) / math.log1p(max_w)
        rendered_edge_color = (
            negative_color
            if include_negative and prey_tsn < 0
            else edge_color
        )
        body.append(
            f'<path d="M{x_pred + 10},{y1:.1f} C{x_pred + 260},{y1:.1f} {x_prey - 260},{y2:.1f} {x_prey - 10},{y2:.1f}" '
            f'fill="none" stroke="{rendered_edge_color}" stroke-opacity="{edge_opacity}" stroke-width="{sw:.2f}" '
            f'marker-end="url(#flow-arrow)"/>'
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
        color = negative_color if is_non_trophic else PREY_COLOR
        body.append(f'<circle cx="{x_prey}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" stroke="#FFFFFF" stroke-width="1.5"/>')
        body.append(f'<text x="{x_prey + 18}" y="{y + 4:.1f}" class="small">{esc(truncate(prey, 36))}</text>')
    legend_y = height - 24
    body.append(f'<circle cx="30" cy="{legend_y - 4}" r="6" fill="{PRED_COLOR}"/><text x="44" y="{legend_y}" class="small">predator</text>')
    body.append(f'<circle cx="130" cy="{legend_y - 4}" r="6" fill="{PREY_COLOR}"/><text x="144" y="{legend_y}" class="small">prey</text>')
    if include_negative:
        body.append(f'<circle cx="220" cy="{legend_y - 4}" r="6" fill="{negative_color}"/><text x="234" y="{legend_y}" class="small">negative TSN/non-trophic category</text>')
    write_svg(name, width, height, "\n".join(body))


def numeric_distribution_grid(
    name="numeric_variable_distributions.svg",
    title="Distributions of key continuous variables",
    subtitle="Counts are aggregated into 24 bins; heavy-tailed quantities use signed log10(1 + x).",
    width=1200,
    height=900,
    panel_width=570,
    panel_height=250,
    palette=None,
):
    histogram_rows = read_rows("numeric_variable_histograms")
    summaries = {row["variable"]: row for row in read_rows("numeric_variable_summary")}
    grouped = defaultdict(list)
    for row in histogram_rows:
        grouped[row["variable"]].append(row)
    variables = list(summaries)
    palette = palette or PALETTE
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for index, variable in enumerate(variables):
        rows = sorted(grouped[variable], key=lambda row: as_int(row["bin_index"]))
        summary = summaries[variable]
        col = index % 2
        row_index = index // 2
        x0 = 30 + col * panel_width
        y0 = 88 + row_index * panel_height
        plot_x, plot_y = x0 + 52, y0 + 48
        plot_w, plot_h = 470, 130
        max_count = max((as_float(row["row_count"]) for row in rows), default=1)
        bar_w = plot_w / max(1, len(rows))
        body.append(f'<text x="{x0}" y="{y0 + 18}" font-weight="700">{esc(variable)}</text>')
        transform = rows[0]["transform"] if rows else "identity"
        transform_label = "linear" if transform == "identity" else "log10(1 + x)"
        body.append(
            f'<text x="{x0}" y="{y0 + 38}" class="small">Unit: {esc(summary["unit"])}; '
            f'{transform_label}; n={fmt_int(summary["non_null_rows"])}</text>'
        )
        body.append(
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" '
            f'y2="{plot_y + plot_h}" stroke="{GRID_COLOR}"/>'
        )
        for bin_index, item in enumerate(rows):
            count = as_float(item["row_count"])
            bar_height = 0 if max_count == 0 else plot_h * count / max_count
            x = plot_x + bin_index * bar_w
            y = plot_y + plot_h - bar_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_w - 1):.1f}" '
                f'height="{bar_height:.1f}" fill="{palette[index % len(palette)]}"/>'
            )
        if rows:
            body.append(
                f'<text x="{plot_x}" y="{plot_y + plot_h + 18}" class="small">'
                f'{fmt_num(rows[0]["lower_transformed"])}</text>'
            )
            body.append(
                f'<text x="{plot_x + plot_w}" y="{plot_y + plot_h + 18}" text-anchor="end" class="small">'
                f'{fmt_num(rows[-1]["upper_transformed"])}</text>'
            )
        body.append(
            f'<text x="{x0}" y="{plot_y + plot_h + 42}" class="small">'
            f'median={fmt_num(summary["median"])}; p99={fmt_num(summary["p99"])}; '
            f'max={fmt_num(summary["max"])}</text>'
        )
    write_svg(name, width, height, "\n".join(body))


def histogram(values, bins=18):
    if not values:
        return 0.0, 1.0, [0] * bins
    minimum = min(values)
    maximum = max(values)
    counts = [0] * bins
    if minimum == maximum:
        counts[0] = len(values)
        return minimum, maximum, counts
    for value in values:
        index = int((value - minimum) / (maximum - minimum) * bins)
        counts[max(0, min(bins - 1, index))] += 1
    return minimum, maximum, counts


def network_metric_distribution_grid(
    metrics,
    name="network_metric_distributions.svg",
    title="Distribution of sea-decade network properties",
    subtitle=None,
    definitions=None,
    width=1100,
    height=650,
    panel_width=520,
    panel_height=260,
    bins=18,
):
    definitions = definitions or [
        ("taxon_nodes", "Taxon nodes (S)", PRED_COLOR),
        ("directed_edges", "Directed links (L)", PREY_COLOR),
        ("nonself_directed_density", "Non-self directed density", NON_TROPHIC_COLOR),
        ("links_per_taxon", "Links per taxon (L/S)", BOTH_ROLE_COLOR),
    ]
    subtitle = subtitle or (
        f"{len(metrics)} complete positive-prey-TSN networks; "
        "values are not effort-standardised."
    )
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for index, (key, label, color) in enumerate(definitions):
        values = [as_float(row[key]) for row in metrics]
        minimum, maximum, counts = histogram(values, bins=bins)
        col, row_index = index % 2, index // 2
        x0 = 30 + col * panel_width
        y0 = 88 + row_index * panel_height
        plot_x, plot_y = x0 + 48, y0 + 48
        plot_w, plot_h = 420, 145
        max_count = max(counts) or 1
        bar_w = plot_w / len(counts)
        body.append(f'<text x="{x0}" y="{y0 + 20}" font-weight="700">{esc(label)}</text>')
        for bin_index, count in enumerate(counts):
            bar_height = plot_h * count / max_count
            x = plot_x + bin_index * bar_w
            y = plot_y + plot_h - bar_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(1, bar_w - 2):.1f}" '
                f'height="{bar_height:.1f}" fill="{color}"/>'
            )
        body.append(
            f'<line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" '
            f'y2="{plot_y + plot_h}" stroke="{GRID_COLOR}"/>'
        )
        body.append(f'<text x="{plot_x}" y="{plot_y + plot_h + 20}" class="small">{fmt_num(minimum)}</text>')
        body.append(
            f'<text x="{plot_x + plot_w}" y="{plot_y + plot_h + 20}" text-anchor="end" '
            f'class="small">{fmt_num(maximum)}</text>'
        )
    write_svg(name, width, height, "\n".join(body))


def effort_vs_edges_chart(
    metrics,
    representatives,
    name="effort_vs_directed_edges.svg",
    title="Sampling effort and directed-link richness",
    subtitle="Sea-decade networks; both axes use log10(1 + x). Association is descriptive, not causal.",
    width=1000,
    height=600,
    background_point_color="#78909C",
    background_point_opacity=0.42,
    background_point_radius=4,
    representative_point_radius=8,
):
    left, right, top, bottom = 90, 50, 90, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    points = [
        (
            math.log10(1 + as_float(row["hauls"])),
            math.log10(1 + as_float(row["directed_edges"])),
            row,
        )
        for row in metrics
    ]
    min_x = min(point[0] for point in points)
    max_x = max(point[0] for point in points)
    min_y = min(point[1] for point in points)
    max_y = max(point[1] for point in points)

    def scale(value, minimum, maximum, start, span, invert=False):
        fraction = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
        return start + span * (1 - fraction if invert else fraction)

    representative_keys = {
        (row["sea"], as_int(row["decade"])): index
        for index, row in enumerate(representatives)
    }
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{GRID_COLOR}"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{GRID_COLOR}"/>',
    ]
    for x_value, y_value, row in points:
        x = scale(x_value, min_x, max_x, left, plot_w)
        y = scale(y_value, min_y, max_y, top, plot_h, invert=True)
        key = row["sea"], as_int(row["decade"])
        if key in representative_keys:
            color = PALETTE[representative_keys[key]]
            radius, opacity = representative_point_radius, 0.95
        else:
            color = background_point_color
            radius, opacity = background_point_radius, background_point_opacity
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" fill-opacity="{opacity}"/>'
        )
        if key in representative_keys:
            body.append(
                f'<text x="{x + 10:.1f}" y="{y - 8:.1f}" class="small">'
                f'{esc(row["sea"])} {as_int(row["decade"])}</text>'
            )
    body.append(
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle">log10(1 + hauls)</text>'
    )
    body.append(
        f'<text x="24" y="{top + plot_h / 2}" transform="rotate(-90 24,{top + plot_h / 2})" '
        f'text-anchor="middle">log10(1 + directed links)</text>'
    )
    write_svg(name, width, height, "\n".join(body))


def slugify(value):
    text = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    return "_".join(part for part in text.split("_") if part)


def unified_network_backbone(
    name,
    title,
    subtitle,
    full_edges,
    limit=24,
    width=1180,
    minimum_height=720,
    row_spacing=34,
    role_x_positions=None,
    role_colors=None,
    edge_color="#657786",
    edge_opacity=0.32,
    minimum_node_radius=6,
    node_radius_range=8,
    minimum_edge_width=0.8,
    edge_width_range=4.5,
):
    edges = sorted(full_edges, key=lambda edge: edge["prey_records"], reverse=True)[:limit]
    predators = {edge["predator_tsn"] for edge in edges}
    prey = {edge["prey_tsn"] for edge in edges}
    nodes = predators | prey
    labels = {}
    strength = defaultdict(float)
    for edge in edges:
        predator = edge["predator_tsn"]
        resource = edge["prey_tsn"]
        labels.setdefault(predator, edge["predator_name"])
        if len(edge["prey_name"]) > len(labels.get(resource, "")):
            labels[resource] = edge["prey_name"]
        weight = as_float(edge["prey_records"])
        strength[predator] += weight
        strength[resource] += weight

    roles = {
        node: (
            "both"
            if node in predators and node in prey
            else "predator"
            if node in predators
            else "prey"
        )
        for node in nodes
    }
    role_nodes = {
        role: sorted(
            [node for node in nodes if roles[node] == role],
            key=lambda node: (-strength[node], labels.get(node, node), node),
        )
        for role in ["predator", "both", "prey"]
    }
    max_role_count = max((len(values) for values in role_nodes.values()), default=1)
    height = max(minimum_height, 150 + max_role_count * row_spacing)
    top, bottom = 108, 80
    x_by_role = role_x_positions or {"predator": 190, "both": 590, "prey": 990}
    color_by_role = role_colors or {
        "predator": PRED_COLOR,
        "both": BOTH_ROLE_COLOR,
        "prey": PREY_COLOR,
    }

    def positions(values):
        if len(values) == 1:
            return {values[0]: top + (height - top - bottom) / 2}
        span = height - top - bottom
        return {
            node: top + index * span / max(1, len(values) - 1)
            for index, node in enumerate(values)
        }

    y_by_node = {}
    for values in role_nodes.values():
        y_by_node.update(positions(values))
    max_strength = max(strength.values(), default=1)
    radius = {
        node: minimum_node_radius
        + node_radius_range * math.sqrt(strength[node] / max_strength)
        for node in nodes
    }
    max_weight = max((as_float(edge["prey_records"]) for edge in edges), default=1)
    body = [
        '<defs><marker id="network-arrow" markerUnits="userSpaceOnUse" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto">'
        f'<path d="M0,0 L0,8 L8,4 z" fill="{edge_color}"/></marker></defs>',
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
        f'<text x="{x_by_role["predator"]}" y="88" text-anchor="middle" class="small">Predator-only TSN</text>',
        f'<text x="{x_by_role["both"]}" y="88" text-anchor="middle" class="small">TSN in both roles</text>',
        f'<text x="{x_by_role["prey"]}" y="88" text-anchor="middle" class="small">Prey-only TSN</text>',
    ]
    for edge in edges:
        source = edge["predator_tsn"]
        target = edge["prey_tsn"]
        source_x = x_by_role[roles[source]]
        target_x = x_by_role[roles[target]]
        source_y = y_by_node[source]
        target_y = y_by_node[target]
        width_scale = (
            minimum_edge_width
            + edge_width_range
            * math.log1p(as_float(edge["prey_records"]))
            / math.log1p(max_weight)
        )
        if source == target:
            path = (
                f'M{source_x + radius[source]:.1f},{source_y:.1f} '
                f'C{source_x + 95:.1f},{source_y - 70:.1f} '
                f'{source_x + 95:.1f},{source_y + 70:.1f} '
                f'{source_x + radius[source]:.1f},{source_y + 4:.1f}'
            )
        elif source_x == target_x:
            direction = 1 if target_y >= source_y else -1
            path = (
                f'M{source_x + radius[source]:.1f},{source_y:.1f} '
                f'C{source_x + 120:.1f},{source_y + 30 * direction:.1f} '
                f'{target_x + 120:.1f},{target_y - 30 * direction:.1f} '
                f'{target_x + radius[target]:.1f},{target_y:.1f}'
            )
        else:
            direction = 1 if target_x > source_x else -1
            start_x = source_x + direction * radius[source]
            end_x = target_x - direction * radius[target]
            control_1 = start_x + direction * abs(end_x - start_x) * 0.38
            control_2 = end_x - direction * abs(end_x - start_x) * 0.38
            path = (
                f'M{start_x:.1f},{source_y:.1f} '
                f'C{control_1:.1f},{source_y:.1f} '
                f'{control_2:.1f},{target_y:.1f} '
                f'{end_x:.1f},{target_y:.1f}'
            )
        body.append(
            f'<path d="{path}" fill="none" stroke="{edge_color}" stroke-opacity="{edge_opacity}" '
            f'stroke-width="{width_scale:.2f}" marker-end="url(#network-arrow)"/>'
        )
    for node in sorted(
        nodes,
        key=lambda value: (
            roles[value],
            -strength[value],
            labels.get(value, value),
            value,
        ),
    ):
        role = roles[node]
        x = x_by_role[role]
        y = y_by_node[node]
        body.append(
            f'<circle cx="{x}" cy="{y:.1f}" r="{radius[node]:.1f}" '
            f'fill="{color_by_role[role]}" stroke="#FFFFFF" stroke-width="1.5"/>'
        )
        if role == "predator":
            text_x, anchor = x - 18, "end"
        else:
            text_x, anchor = x + 18, "start"
        body.append(
            f'<text x="{text_x}" y="{y + 4:.1f}" text-anchor="{anchor}" class="small">'
            f'{esc(truncate(labels.get(node, node), 34))} [TSN {esc(node)}]</text>'
        )
    legend_y = height - 30
    legend = [
        (PRED_COLOR, "predator-only"),
        (BOTH_ROLE_COLOR, "both roles"),
        (PREY_COLOR, "prey-only"),
    ]
    for index, (color, label) in enumerate(legend):
        x = 30 + index * 140
        body.append(f'<circle cx="{x}" cy="{legend_y - 4}" r="6" fill="{color}"/>')
        body.append(f'<text x="{x + 14}" y="{legend_y}" class="small">{label}</text>')
    body.append(
        f'<text x="460" y="{legend_y}" class="small">arrow: predator → consumed prey; '
        f'edge width and node size: prey-record support</text>'
    )
    write_svg(name, width, height, "\n".join(body))


def representative_metric_comparison(
    representatives,
    name="representative_network_metric_comparison.svg",
    title="Structural comparison of representative networks",
    subtitle="Bar lengths are normalised within each metric; labels give full-network values.",
    definitions=None,
    width=1100,
    height=660,
    colors=None,
    legend_start_x=None,
    legend_item_width=265,
):
    definitions = definitions or [
        ("taxon_nodes", "Taxon nodes (S)", lambda value: fmt_int(value)),
        ("directed_edges", "Directed links (L)", lambda value: fmt_int(value)),
        ("nonself_directed_density", "Non-self density", lambda value: f"{as_float(value):.4f}"),
        ("links_per_taxon", "Links per taxon (L/S)", lambda value: fmt_num(value)),
    ]
    left, top = 250, 110
    plot_w = 680
    group_h = 125
    colors = colors or PALETTE[: len(representatives)]
    if legend_start_x is None:
        legend_start_x = max(
            30,
            (width - legend_item_width * len(representatives)) / 2,
        )
    body = [
        f'<text x="30" y="34" class="title">{esc(title)}</text>',
        f'<text x="30" y="58" class="subtitle">{esc(subtitle)}</text>',
    ]
    for index, row in enumerate(representatives):
        x = legend_start_x + index * legend_item_width
        body.append(f'<rect x="{x}" y="76" width="12" height="12" fill="{colors[index]}"/>')
        body.append(
            f'<text x="{x + 18}" y="87" class="small">'
            f'{esc(row["complexity_tier"])}: {esc(row["sea"])} {as_int(row["decade"])}</text>'
        )
    for metric_index, (key, label, formatter) in enumerate(definitions):
        y0 = top + metric_index * group_h
        maximum = max(as_float(row[key]) for row in representatives) or 1
        body.append(f'<text x="{left - 18}" y="{y0 + 42}" text-anchor="end">{esc(label)}</text>')
        for rep_index, row in enumerate(representatives):
            y = y0 + rep_index * 28
            value = as_float(row[key])
            bar_width = plot_w * value / maximum
            body.append(
                f'<rect x="{left}" y="{y}" width="{bar_width:.1f}" height="18" '
                f'rx="3" fill="{colors[rep_index]}"/>'
            )
            body.append(
                f'<text x="{left + bar_width + 8:.1f}" y="{y + 14}" class="small">'
                f'{formatter(value)}</text>'
            )
    write_svg(name, width, height, "\n".join(body))


def build_figures(networks, metrics, representatives):
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
    numeric_distribution_grid()
    network_metric_distribution_grid(metrics)
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
    effort_vs_edges_chart(metrics, representatives)
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
    for row in representatives:
        key = row["sea"], as_int(row["decade"])
        figure_name = (
            f"network_representative_{as_int(row['selection_rank'])}_"
            f"{slugify(row['sea'])}_{as_int(row['decade'])}.svg"
        )
        row["figure_name"] = figure_name
        unified_network_backbone(
            figure_name,
            f"{row['complexity_tier'].capitalize()}-complexity network: "
            f"{row['sea']} {as_int(row['decade'])}",
            (
                f"Readable top-24 link backbone; full graph has S={as_int(row['taxon_nodes'])}, "
                f"L={as_int(row['directed_edges'])}, {as_int(row['hauls'])} hauls."
            ),
            networks[key],
            limit=24,
        )
    representative_metric_comparison(representatives)


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


def svg_code_cell(source, figure_name):
    svg_path = FIG_DIR / figure_name
    output = {
        "data": {
            "image/svg+xml": svg_path.read_text(encoding="utf-8"),
            "text/plain": f"<SVG: {figure_name}>",
        },
        "metadata": {},
        "output_type": "display_data",
    }
    return code_cell(source, outputs=[output])


def figure_md(path, caption):
    return f'<img src="{path}" alt="{esc(caption)}" style="max-width:100%; height:auto;">\n\n*{caption}*'


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _build_static_notebook_legacy(representatives):
    representative_rows = [
        [
            row["complexity_tier"].capitalize(),
            row["sea"],
            as_int(row["decade"]),
            fmt_int(row["hauls"]),
            fmt_int(row["taxon_nodes"]),
            fmt_int(row["directed_edges"]),
            f"{as_float(row['nonself_directed_density']):.4f}",
            fmt_num(row["links_per_taxon"]),
            f"{as_float(row['largest_component_pct']):.1f}%",
        ]
        for row in representatives
    ]
    cells = [
        markdown_cell(
            "# DAPSTOM 6.4 Visual EDA and Representative Networks\n\n"
            "This notebook complements `DAPSTOM_EDA.ipynb` with visual summaries, continuous-variable "
            "distributions, complete-network metric distributions and reproducibly selected individual networks. "
            "Every SVG is generated from the derived CSV tables; no figure is edited by hand."
        ),
        markdown_cell(
            "## How to regenerate\n\n"
            "From the repo root:\n\n"
            "```bash\n"
            "bash data/processed/dapstom_eda/tools/run_extractor.sh\n"
            "python3 data/processed/dapstom_eda/tools/derive_network_metrics.py\n"
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
        markdown_cell(
            "Temporal coverage is long but highly uneven; apparent temporal network change must be compared with effort."
        ),
        markdown_cell("## 3. Spatial Coverage\n\n" + figure_md("figures/hauls_by_sea.svg", "Top sea-level groups by haul count.")),
        markdown_cell(figure_md("figures/spatial_resolution_coverage.svg", "Availability of point coordinates, ICES rectangle, ICES division, and sea.")),
        markdown_cell("## 4. Sampling Unit: Pooled vs Individual\n\n" + figure_md("figures/pooled_rows_vs_stomachs.svg", "Pooled rows are fewer but represent many stomachs.")),
        markdown_cell(
            "Pooled records are not exchangeable with individual-stomach rows. Figures and models should report "
            "hauls and represented stomachs rather than treating row count alone as effort."
        ),
        markdown_cell("## 5. Missingness\n\n" + figure_md("figures/critical_missingness.svg", "Missingness in fields needed for spatial, temporal, taxonomy and evidence-aware modelling.")),
        markdown_cell(
            "## 6. Continuous-variable Distributions and Outlier Context\n\n"
            + figure_md(
                "figures/numeric_variable_distributions.svg",
                "Key numerical distributions; heavy-tailed effort and diet quantities use log10(1 + x).",
            )
        ),
        markdown_cell(
            "The plots retain extreme values. Tukey-IQR flags in the main notebook are review prompts, not automatic "
            "deletions: pooled samples and rare dietary events can legitimately generate long tails."
        ),
        markdown_cell("## 7. Dominant Taxa and Non-standard Categories\n\n" + figure_md("figures/top_predators.svg", "Most common predator records.")),
        markdown_cell(figure_md("figures/negative_tsn_categories.svg", "Negative prey TSN categories that should usually be separated from ordinary trophic edges.")),
        markdown_cell(
            "The resolved-taxon primary graph requires `prey_tsn > 0`. Empty and non-trophic codes remain useful "
            "sampling/QC information; unresolved trophic codes (-99913 to -99915) should also be tested in a separate "
            "sensitivity analysis rather than silently treated as identified taxa."
        ),
        markdown_cell(
            "## 8. Network-property Distributions\n\n"
            + figure_md(
                "figures/network_metric_distributions.svg",
                "Distributions of S, L, non-self density and L/S across complete positive-prey-TSN sea-decade graphs.",
            )
        ),
        markdown_cell(
            "## 9. Spatial and Temporal Network Coverage\n\n"
            + figure_md(
                "figures/sea_decade_edge_heatmap.svg",
                "Resolved predator-prey name-pair richness for every available sea-decade stratum.",
            )
        ),
        markdown_cell(
            "This heatmap reports **link richness**, not density. All positive-prey-TSN strata are included; missing "
            "combinations are not created by truncating a top-100 table."
        ),
        markdown_cell(
            figure_md(
                "figures/effort_vs_directed_edges.svg",
                "Sampling effort versus directed TSN-link richness; representative networks are highlighted.",
            )
        ),
        markdown_cell(
            "The effort-link association is descriptive. Network comparisons require effort adjustment, minimum "
            "eligibility thresholds and sensitivity analyses rather than interpreting raw L as ecological complexity alone."
        ),
        markdown_cell(
            "## 10. Global Filter Diagnostic\n\n"
            "These two global role-flow diagrams are quality-control views, not the representative individual "
            "sea-decade networks. They show why raw non-positive categories cannot be mixed into the primary graph."
        ),
        markdown_cell(figure_md("figures/network_top_raw_edges.svg", "Raw top links, including non-trophic and negative-TSN categories.")),
        markdown_cell(figure_md("figures/network_positive_prey_tsn_edges.svg", "Global top links after requiring a positive prey TSN.")),
        markdown_cell(
            "## 11. Representative Individual Networks\n\n"
            "Selection rule: among strata with at least **30 hauls**, **10 predator taxa** and **100 prey taxa**, "
            "select high, median and low directed-link complexity while preferring different seas.\n\n"
            + md_table(
                [
                    "Tier",
                    "Sea",
                    "Decade",
                    "Hauls",
                    "S",
                    "L",
                    "Non-self density",
                    "L/S",
                    "Largest component",
                ],
                representative_rows,
            )
        ),
        markdown_cell(
            "### How to read the network figures\n\n"
            "- **Node:** one unified positive TSN. A taxon occurring as both predator and prey is drawn once in the centre.\n"
            "- **Colour:** blue = predator-only, green = prey-only, purple = observed in both roles.\n"
            "- **Arrow:** `predator → consumed prey`; ecological energy flow is the reverse.\n"
            "- **Edge width:** prey-record support for that directed pair within the sea-decade stratum.\n"
            "- **Node size:** total prey-record support incident on the node in the displayed backbone.\n"
            "- **Displayed structure:** the 24 most supported links for legibility. Titles and comparison tables report "
            "metrics from the complete graph, not only the displayed backbone.\n"
            "- **Available edge attributes:** `prey_records`, distinct-haul support, summed `min_num` and summed `cpw`; "
            "pooling, effort and provenance remain available for later modelling."
        ),
    ]
    for row in representatives:
        cells.append(
            markdown_cell(
                f"### {row['complexity_tier'].capitalize()} complexity — "
                f"{row['sea']} {as_int(row['decade'])}\n\n"
                + figure_md(
                    f"figures/{row['figure_name']}",
                    (
                        f"Top-supported backbone of the {row['sea']} {as_int(row['decade'])} network; "
                        f"complete graph S={as_int(row['taxon_nodes'])}, L={as_int(row['directed_edges'])}, "
                        f"density={as_float(row['nonself_directed_density']):.4f}."
                    ),
                )
            )
        )
    densest = max(representatives, key=lambda row: row["nonself_directed_density"])
    concentrated = max(
        representatives, key=lambda row: row["top_10_edge_record_share_pct"]
    )
    cells.extend(
        [
            markdown_cell(
                "## 12. Structural Comparison\n\n"
                + figure_md(
                    "figures/representative_network_metric_comparison.svg",
                    "Full-graph comparison of S, L, non-self density and L/S for the selected examples.",
                )
            ),
            markdown_cell(
                f"**{densest['sea']} {as_int(densest['decade'])}** is densest among the selected examples "
                f"({as_float(densest['nonself_directed_density']):.4f}); "
                f"**{concentrated['sea']} {as_int(concentrated['decade'])}** has the largest top-10 link "
                f"record share ({as_float(concentrated['top_10_edge_record_share_pct']):.1f}%). "
                "Differences in S, L, density, L/S and concentration demonstrate that link count alone does not "
                "fully describe structure. They remain hypotheses for effort- and provenance-aware analysis, not "
                "causal ecological conclusions."
            ),
            markdown_cell(
                "## Interpretation Boundary\n\n"
                "The figures establish representative examples and reproducible structural descriptors. Formal "
                "ecological comparisons must follow the preprocessing, grouped statistical analyses and validation "
                "sequence defined in `DAPSTOM_EDA.ipynb`."
            ),
        ]
    )
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


def build_notebook(representatives):
    for row in representatives:
        row.setdefault(
            "figure_name",
            (
                f"network_representative_{as_int(row['selection_rank'])}_"
                f"{slugify(row['sea'])}_{as_int(row['decade'])}.svg"
            ),
        )

    representative_rows = [
        [
            row["complexity_tier"].capitalize(),
            row["sea"],
            as_int(row["decade"]),
            fmt_int(row["hauls"]),
            fmt_int(row["taxon_nodes"]),
            fmt_int(row["directed_edges"]),
            f"{as_float(row['nonself_directed_density']):.4f}",
            fmt_num(row["links_per_taxon"]),
            f"{as_float(row['largest_component_pct']):.1f}%",
        ]
        for row in representatives
    ]
    cells = []

    def add_figure(heading, explanation, source, figure_name):
        cells.append(markdown_cell(f"{heading}\n\n{explanation}"))
        cells.append(svg_code_cell(source.strip() + "\n", figure_name))

    cells.extend(
        [
            markdown_cell(
                "# DAPSTOM 6.4 Visual EDA and Representative Networks\n\n"
                "This is an executable visual notebook. Each figure has a code cell with editable titles, colours, "
                "limits and layout parameters. Running one figure cell regenerates only its named SVG; running all "
                "cells reproduces the complete visual EDA from the derived CSV tables."
            ),
            markdown_cell(
                "## Reproducibility and customisation workflow\n\n"
                "The source Access database is not queried or modified by the plotting cells. Data extraction remains "
                "in `tools/DapstomEdaExtractor.java`; network derivation remains in "
                "`tools/derive_network_metrics.py`; reusable SVG drawing functions remain in "
                "`tools/build_visual_notebook.py`.\n\n"
                "To refresh every derived table before running this notebook:\n\n"
                "```bash\n"
                "bash data/processed/dapstom_eda/tools/run_extractor.sh\n"
                "python3 data/processed/dapstom_eda/tools/derive_network_metrics.py\n"
                "```\n\n"
                "After that, use **Kernel → Restart & Run All**, or edit and run a single figure cell."
            ),
            code_cell(
                """from pathlib import Path
import sys

try:
    from IPython.display import SVG, display
except ImportError:
    SVG = None
    display = None


def locate_eda_directory():
    current = Path.cwd().resolve()
    for root in [current, *current.parents]:
        if root.name == "dapstom_eda" and (root / "tables").exists():
            return root
        candidate = root / "data" / "processed" / "dapstom_eda"
        if (candidate / "tables").exists():
            return candidate
    raise FileNotFoundError("Could not locate data/processed/dapstom_eda")


EDA_DIR = locate_eda_directory()
TABLE_DIR = EDA_DIR / "tables"
FIG_DIR = EDA_DIR / "figures"
TOOLS_DIR = EDA_DIR / "tools"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_visual_notebook as viz
from derive_network_metrics import derive

viz.TABLE_DIR = TABLE_DIR
viz.FIG_DIR = FIG_DIR


def show_svg(figure_name):
    figure_path = FIG_DIR / figure_name
    if SVG is None:
        return figure_path
    display(SVG(filename=str(figure_path)))


EDA_DIR
"""
            ),
            markdown_cell(
                "### Global visual style\n\n"
                "Change this cell to update the shared palette, trophic-role colours, grid, text and background. "
                "Rerun the desired figure cells afterwards."
            ),
            code_cell(
                """STYLE = {
    "palette": ["#36648B", "#4C8C5A", "#C46A2D", "#7E5AA2", "#A23E48", "#548C8C"],
    "predator": "#36648B",
    "prey": "#4C8C5A",
    "non_trophic": "#C46A2D",
    "both_roles": "#7E5AA2",
    "grid": "#D8DEE4",
    "text": "#1F2933",
    "muted_text": "#52616B",
    "background": "#FFFFFF",
    "font_family": '-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif',
    "base_font_size": 13,
    "title_font_size": 22,
    "subtitle_font_size": 13,
    "small_font_size": 11,
}

viz.PALETTE = STYLE["palette"]
viz.PRED_COLOR = STYLE["predator"]
viz.PREY_COLOR = STYLE["prey"]
viz.NON_TROPHIC_COLOR = STYLE["non_trophic"]
viz.BOTH_ROLE_COLOR = STYLE["both_roles"]
viz.GRID_COLOR = STYLE["grid"]
viz.TEXT_COLOR = STYLE["text"]
viz.MUTED_TEXT = STYLE["muted_text"]
viz.BG = STYLE["background"]
viz.FONT_FAMILY = STYLE["font_family"]
viz.BASE_FONT_SIZE = STYLE["base_font_size"]
viz.TITLE_FONT_SIZE = STYLE["title_font_size"]
viz.SUBTITLE_FONT_SIZE = STYLE["subtitle_font_size"]
viz.SMALL_FONT_SIZE = STYLE["small_font_size"]
"""
            ),
        ]
    )

    add_figure(
        "## 1. Database Scale",
        "Source: `table_row_counts.csv`. Adjust `LIMIT`, colour, width, label space or row height.",
        """FIGURE_NAME = "table_row_counts.svg"
LIMIT = 10

rows = viz.read_rows("table_row_counts")
viz.horizontal_bar_chart(
    name=FIGURE_NAME,
    title="DAPSTOM table sizes",
    subtitle="Rows in user tables exposed by the Access database.",
    rows=rows,
    value_key="row_count",
    label_key="table_name",
    limit=LIMIT,
    color=STYLE["predator"],
    width=980,
    row_height=34,
    left=270,
    right=80,
)
show_svg(FIGURE_NAME)
""",
        "table_row_counts.svg",
    )

    add_figure(
        "## 2. Temporal Coverage",
        "Source: `haul_temporal_by_decade.csv`. Bar colour, canvas dimensions and spacing are editable.",
        """FIGURE_NAME = "hauls_by_decade.svg"

rows = viz.read_rows("haul_temporal_by_decade")
viz.vertical_bar_chart(
    name=FIGURE_NAME,
    title="Haul records by decade",
    subtitle="Temporal spread in the HAUL table; early historical data are sparse.",
    rows=rows,
    value_key="haul_rows",
    label_key="decade",
    color=STYLE["prey"],
    width=1040,
    height=520,
    bar_gap=5,
)
show_svg(FIGURE_NAME)
""",
        "hauls_by_decade.svg",
    )
    cells.append(
        markdown_cell(
            "Temporal coverage is long but highly uneven; apparent temporal network change must be compared with effort."
        )
    )

    add_figure(
        "## 3. Spatial Coverage — hauls by sea",
        "Source: `haul_spatial_by_sea.csv`. `LIMIT` controls the number of seas displayed.",
        """FIGURE_NAME = "hauls_by_sea.svg"
LIMIT = 12

rows = viz.read_rows("haul_spatial_by_sea")
viz.horizontal_bar_chart(
    name=FIGURE_NAME,
    title="Haul records by sea",
    subtitle="Top sea-level groups by haul count.",
    rows=rows,
    value_key="haul_rows",
    label_key="sea",
    limit=LIMIT,
    color=STYLE["both_roles"],
    width=980,
    row_height=34,
    left=270,
    right=80,
)
show_svg(FIGURE_NAME)
""",
        "hauls_by_sea.svg",
    )
    add_figure(
        "### Spatial resolution",
        "Source: `haul_spatial_resolution.csv`. Edit the colour sequence, canvas and bar width.",
        """FIGURE_NAME = "spatial_resolution_coverage.svg"

viz.spatial_coverage_chart(
    name=FIGURE_NAME,
    title="Spatial resolution available by haul",
    subtitle="Point coordinates are partial; ICES division and sea are complete.",
    width=900,
    height=360,
    left=210,
    top=92,
    plot_width=560,
    row_height=42,
    colors=STYLE["palette"],
)
show_svg(FIGURE_NAME)
""",
        "spatial_resolution_coverage.svg",
    )

    add_figure(
        "## 4. Sampling Unit: Pooled vs Individual",
        "Source: `predator_pooling_summary.csv`. Category colours, labels and geometry are visible below.",
        """FIGURE_NAME = "pooled_rows_vs_stomachs.svg"
POOL_COLOURS = {
    "n": STYLE["predator"],
    "y": STYLE["non_trophic"],
    "": "#8B949E",
}
POOL_LABELS = {
    "n": "individual rows",
    "y": "pooled rows",
    "": "missing",
}

viz.pooled_comparison_chart(
    name=FIGURE_NAME,
    title="Pooled records change the sampling unit",
    subtitle="Pooled rows are few, but represent many stomachs.",
    width=920,
    height=300,
    bar_width=580,
    bar_height=34,
    colors=POOL_COLOURS,
    labels=POOL_LABELS,
)
show_svg(FIGURE_NAME)
""",
        "pooled_rows_vs_stomachs.svg",
    )
    cells.append(
        markdown_cell(
            "Pooled records are not exchangeable with individual-stomach rows. Report hauls and represented stomachs "
            "rather than treating row count alone as effort."
        )
    )

    add_figure(
        "## 5. Missingness",
        "Source: `critical_field_missingness.csv`. `LIMIT` controls the number of fields shown.",
        """FIGURE_NAME = "critical_missingness.svg"
LIMIT = 10

viz.missingness_chart(
    name=FIGURE_NAME,
    title="Critical-field missingness",
    subtitle="Null and blank counts in fields needed for spatial, temporal, and evidence handling.",
    limit=LIMIT,
    color="#A23E48",
)
show_svg(FIGURE_NAME)
""",
        "critical_missingness.svg",
    )

    add_figure(
        "## 6. Continuous-variable Distributions and Outlier Context",
        "Sources: `numeric_variable_histograms.csv` and `numeric_variable_summary.csv`. Edit panel dimensions, "
        "titles and palette.",
        """FIGURE_NAME = "numeric_variable_distributions.svg"

viz.numeric_distribution_grid(
    name=FIGURE_NAME,
    title="Distributions of key continuous variables",
    subtitle="Counts are aggregated into 24 bins; heavy-tailed quantities use signed log10(1 + x).",
    width=1200,
    height=900,
    panel_width=570,
    panel_height=250,
    palette=STYLE["palette"],
)
show_svg(FIGURE_NAME)
""",
        "numeric_variable_distributions.svg",
    )
    cells.append(
        markdown_cell(
            "Extreme values are retained. Tukey-IQR flags in the main notebook are review prompts, not automatic "
            "deletions: pooled samples and rare dietary events can legitimately generate long tails."
        )
    )

    add_figure(
        "## 7. Dominant Taxa",
        "Source: `top_predators.csv`. Edit `LIMIT`, colour and label space.",
        """FIGURE_NAME = "top_predators.svg"
LIMIT = 12

rows = viz.read_rows("top_predators")
viz.horizontal_bar_chart(
    name=FIGURE_NAME,
    title="Most common predator records",
    subtitle="Top predators by row count in the PREDATOR table.",
    rows=rows,
    value_key="predator_rows",
    label_key="predator_name",
    limit=LIMIT,
    color=STYLE["predator"],
    width=980,
    row_height=34,
    left=270,
    right=80,
)
show_svg(FIGURE_NAME)
""",
        "top_predators.svg",
    )
    add_figure(
        "### Non-standard TSN categories",
        "Source: `negative_tsn_prey_categories.csv`. This is a QC summary rather than a trophic network.",
        """FIGURE_NAME = "negative_tsn_categories.svg"
LIMIT = 12

rows = viz.read_rows("negative_tsn_prey_categories")
viz.horizontal_bar_chart(
    name=FIGURE_NAME,
    title="Negative TSN prey categories",
    subtitle="Empty, unknown, digested, and other non-standard categories to model separately.",
    rows=rows,
    value_key="prey_rows",
    label_key="prey_name",
    limit=LIMIT,
    color=STYLE["non_trophic"],
    width=980,
    row_height=34,
    left=270,
    right=80,
)
show_svg(FIGURE_NAME)
""",
        "negative_tsn_categories.svg",
    )
    cells.append(
        markdown_cell(
            "The resolved-taxon primary graph requires `prey_tsn > 0`. Empty and non-trophic codes remain useful "
            "sampling/QC information; unresolved trophic codes (-99913 to -99915) belong in a sensitivity analysis."
        )
    )

    cells.extend(
        [
            markdown_cell(
                "## Derive complete sea × decade networks\n\n"
                "This reconstructs unified TSN edges, complete-network metrics and the deterministic representative "
                "selection from derived CSV tables. It does not query or change the Access database."
            ),
            code_cell(
                """networks, network_metrics, representatives = derive()
{
    "networks": len(networks),
    "metric_rows": len(network_metrics),
    "representatives": [
        (row["complexity_tier"], row["sea"], row["decade"])
        for row in representatives
    ],
}
"""
            ),
        ]
    )

    add_figure(
        "## 8. Network-property Distributions",
        "Change `BINS` or `METRICS` to adjust the number and definition of panels.",
        """FIGURE_NAME = "network_metric_distributions.svg"
BINS = 18
METRICS = [
    ("taxon_nodes", "Taxon nodes (S)", STYLE["predator"]),
    ("directed_edges", "Directed links (L)", STYLE["prey"]),
    ("nonself_directed_density", "Non-self directed density", STYLE["non_trophic"]),
    ("links_per_taxon", "Links per taxon (L/S)", STYLE["both_roles"]),
]

viz.network_metric_distribution_grid(
    network_metrics,
    name=FIGURE_NAME,
    title="Distribution of sea-decade network properties",
    subtitle=f"{len(network_metrics)} complete positive-prey-TSN networks; values are not effort-standardised.",
    definitions=METRICS,
    width=1100,
    height=650,
    panel_width=520,
    panel_height=260,
    bins=BINS,
)
show_svg(FIGURE_NAME)
""",
        "network_metric_distributions.svg",
    )

    add_figure(
        "## 9. Spatial and Temporal Network Coverage",
        "Source: `edge_pairs_by_sea_decade_positive_prey_tsn.csv`. Edit sea count, cell geometry, colour scale and "
        "label threshold.",
        """FIGURE_NAME = "sea_decade_edge_heatmap.svg"
SEA_LIMIT = 10

viz.heatmap_sea_decade(
    name=FIGURE_NAME,
    title="Resolved link richness by sea and decade",
    subtitle="All sea-decade strata; unique predator-prey name pairs after requiring prey TSN > 0.",
    sea_limit=SEA_LIMIT,
    cell_width=42,
    cell_height=30,
    low_color="#F3F6F8",
    high_color=STYLE["predator"],
    label_threshold=0.38,
)
show_svg(FIGURE_NAME)
""",
        "sea_decade_edge_heatmap.svg",
    )
    cells.append(
        markdown_cell(
            "This heatmap reports **link richness**, not density. All positive-prey-TSN strata are included."
        )
    )
    add_figure(
        "### Sampling effort versus link richness",
        "Point sizes, opacity, canvas dimensions and colours are editable.",
        """FIGURE_NAME = "effort_vs_directed_edges.svg"

viz.effort_vs_edges_chart(
    network_metrics,
    representatives,
    name=FIGURE_NAME,
    title="Sampling effort and directed-link richness",
    subtitle="Sea-decade networks; both axes use log10(1 + x). Association is descriptive, not causal.",
    width=1000,
    height=600,
    background_point_color="#78909C",
    background_point_opacity=0.42,
    background_point_radius=4,
    representative_point_radius=8,
)
show_svg(FIGURE_NAME)
""",
        "effort_vs_directed_edges.svg",
    )
    cells.append(
        markdown_cell(
            "The effort-link association is descriptive. Comparisons require effort adjustment and sensitivity analyses."
        )
    )

    add_figure(
        "## 10. Global Filter Diagnostic — raw categories",
        "This QC flow includes non-positive categories. `LIMIT` controls the displayed edge backbone.",
        """FIGURE_NAME = "network_top_raw_edges.svg"
LIMIT = 30

rows = viz.read_rows("top_predator_prey_edges")
viz.edge_network(
    name=FIGURE_NAME,
    title="Raw top predator-prey rows",
    subtitle="Includes empty/non-standard prey categories; useful as a cautionary view.",
    rows=rows,
    limit=LIMIT,
    include_negative=True,
    width=1160,
    minimum_height=660,
    row_spacing=25,
    edge_opacity=0.34,
    negative_color=STYLE["non_trophic"],
)
show_svg(FIGURE_NAME)
""",
        "network_top_raw_edges.svg",
    )
    add_figure(
        "### Global positive-TSN diagnostic",
        "This uses the positive-prey-TSN top-edge table. Modify `LIMIT`, spacing or edge opacity.",
        """FIGURE_NAME = "network_positive_prey_tsn_edges.svg"
LIMIT = 42

rows = viz.read_rows("top_predator_prey_edges_positive_prey_tsn")
viz.edge_network(
    name=FIGURE_NAME,
    title="Top predator-prey network after filtering negative prey TSN",
    subtitle="A cleaner first-pass trophic network based on the most frequent positive-TSN prey rows.",
    rows=rows,
    limit=LIMIT,
    include_negative=False,
    width=1160,
    minimum_height=660,
    row_spacing=25,
    edge_opacity=0.34,
)
show_svg(FIGURE_NAME)
""",
        "network_positive_prey_tsn_edges.svg",
    )

    cells.extend(
        [
            markdown_cell(
                "## 11. Representative Individual Networks\n\n"
                "Selection rule: among strata with at least **30 hauls**, **10 predator taxa** and **100 prey taxa**, "
                "select high, median and low directed-link complexity while preferring different seas.\n\n"
                + md_table(
                    [
                        "Tier",
                        "Sea",
                        "Decade",
                        "Hauls",
                        "S",
                        "L",
                        "Non-self density",
                        "L/S",
                        "Largest component",
                    ],
                    representative_rows,
                )
            ),
            markdown_cell(
                "### How to read and customise the network figures\n\n"
                "- **Node:** one unified positive TSN; a taxon appearing in both roles is drawn once.\n"
                "- **Colour:** blue = predator-only, green = prey-only, purple = both roles.\n"
                "- **Arrow:** `predator → consumed prey`; ecological energy flow is the reverse.\n"
                "- **Displayed structure:** the most supported links, controlled by `LIMIT`.\n"
                "- **Editable parameters:** role colours, column positions, node-radius range, edge-width range, "
                "edge opacity, row spacing and canvas dimensions."
            ),
        ]
    )
    for row in representatives:
        figure_name = row["figure_name"]
        sea = row["sea"]
        decade = as_int(row["decade"])
        cells.append(
            markdown_cell(
                f"### {row['complexity_tier'].capitalize()} complexity — {sea} {decade}\n\n"
                f"Complete graph: S={as_int(row['taxon_nodes'])}, L={as_int(row['directed_edges'])}, "
                f"non-self density={as_float(row['nonself_directed_density']):.4f}."
            )
        )
        source = f"""FIGURE_NAME = {figure_name!r}
REPRESENTATIVE_KEY = ({sea!r}, {decade})
LIMIT = 24

representative_row = next(
    row
    for row in representatives
    if (row["sea"], viz.as_int(row["decade"])) == REPRESENTATIVE_KEY
)
viz.unified_network_backbone(
    name=FIGURE_NAME,
    title=(
        representative_row["complexity_tier"].capitalize()
        + "-complexity network: "
        + representative_row["sea"]
        + " "
        + str(viz.as_int(representative_row["decade"]))
    ),
    subtitle=(
        "Readable top-{{}} link backbone; full graph has S={{}}, L={{}}, {{}} hauls."
        .format(
            LIMIT,
            viz.as_int(representative_row["taxon_nodes"]),
            viz.as_int(representative_row["directed_edges"]),
            viz.as_int(representative_row["hauls"]),
        )
    ),
    full_edges=networks[REPRESENTATIVE_KEY],
    limit=LIMIT,
    width=1180,
    minimum_height=720,
    row_spacing=34,
    role_x_positions={{"predator": 190, "both": 590, "prey": 990}},
    role_colors={{
        "predator": STYLE["predator"],
        "both": STYLE["both_roles"],
        "prey": STYLE["prey"],
    }},
    edge_color="#657786",
    edge_opacity=0.32,
    minimum_node_radius=6,
    node_radius_range=8,
    minimum_edge_width=0.8,
    edge_width_range=4.5,
)
show_svg(FIGURE_NAME)
"""
        cells.append(svg_code_cell(source, figure_name))

    add_figure(
        "## 12. Structural Comparison",
        "Complete-graph comparison. Edit `COMPARISON_METRICS` to add, remove or rename rows.",
        """FIGURE_NAME = "representative_network_metric_comparison.svg"
COMPARISON_METRICS = [
    ("taxon_nodes", "Taxon nodes (S)", lambda value: viz.fmt_int(value)),
    ("directed_edges", "Directed links (L)", lambda value: viz.fmt_int(value)),
    (
        "nonself_directed_density",
        "Non-self density",
        lambda value: f"{viz.as_float(value):.4f}",
    ),
    ("links_per_taxon", "Links per taxon (L/S)", lambda value: viz.fmt_num(value)),
]

viz.representative_metric_comparison(
    representatives,
    name=FIGURE_NAME,
    title="Structural comparison of representative networks",
    subtitle="Bar lengths are normalised within each metric; labels give full-network values.",
    definitions=COMPARISON_METRICS,
    width=1100,
    height=660,
    colors=STYLE["palette"][: len(representatives)],
    legend_start_x=None,  # None centres the complete legend group
    legend_item_width=265,
)
show_svg(FIGURE_NAME)
""",
        "representative_network_metric_comparison.svg",
    )

    densest = max(representatives, key=lambda row: row["nonself_directed_density"])
    concentrated = max(
        representatives,
        key=lambda row: row["top_10_edge_record_share_pct"],
    )
    cells.extend(
        [
            markdown_cell(
                f"**{densest['sea']} {as_int(densest['decade'])}** is densest among the selected examples "
                f"({as_float(densest['nonself_directed_density']):.4f}); "
                f"**{concentrated['sea']} {as_int(concentrated['decade'])}** has the largest top-10 link "
                f"record share ({as_float(concentrated['top_10_edge_record_share_pct']):.1f}%). Differences in S, L, "
                "density, L/S and concentration remain hypotheses for effort- and provenance-aware analysis."
            ),
            markdown_cell(
                "## Interpretation Boundary\n\n"
                "Formal ecological comparisons must follow the preprocessing, grouped statistical analyses and "
                "validation sequence in `DAPSTOM_EDA.ipynb`. Stable SVG filenames are preserved so presentation assets "
                "continue to work after visual customisation."
            ),
        ]
    )
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    derived_networks, derived_metrics, selected_representatives = derive()
    build_figures(derived_networks, derived_metrics, selected_representatives)
    build_notebook(selected_representatives)
    print(NOTEBOOK_PATH)
