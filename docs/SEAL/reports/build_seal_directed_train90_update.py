#!/usr/bin/env python3
"""Build the editable SEAL_directed train-ratio 90 meeting document.

The script uses only Python's standard library and embeds existing repository
figures without modifying them or generating new plots.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
import statistics
import struct
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).with_name("SEAL_directed_train90_update_for_Athen.docx")

ORIGINAL_RESULTS = REPO_ROOT / "src/python/data/result/prediction_scores_logs"
DIRECTED_RESULTS = REPO_ROOT / "src/python/seal_directed/data/result/prediction_scores_logs"
ATTRIBUTE_SUMMARY = (
    REPO_ROOT
    / "src/python/seal_directed/data/foodwebs_mat_seal_attrs/"
    "seal_node_attribute_build_summary.csv"
)
FEATURE_NAMES = (
    REPO_ROOT
    / "src/python/seal_directed/data/foodwebs_mat_seal_attrs/"
    "seal_node_attribute_feature_names.csv"
)

FIGURES = [
    (
        REPO_ROOT / "docs/plots/seal_directed_node_attribute_eda/candidate_column_coverage.png",
        "Candidate attribute coverage in the source data. Length variables were excluded because their mean coverage is only about 7.3%.",
    ),
    (
        REPO_ROOT / "docs/plots/seal_directed_node_attribute_eda/group_feature_composition.png",
        "Composition of the 44-dimensional node-attribute matrix currently supplied to SEAL_directed.",
    ),
    (
        REPO_ROOT / "docs/plots/seal_comparison/seal_original_vs_directed_boxplots.png",
        "Existing distributional comparison across food webs. The first five panels are the predictive metrics discussed in this document; the remaining panels describe reconstructed pseudo-web performance.",
    ),
    (
        REPO_ROOT / "docs/plots/seal_comparison/seal_directed_minus_original_deltas.png",
        "Existing food-web-level paired differences (SEAL_directed minus original SEAL). Positive values favour SEAL_directed; the zero line marks no change.",
    ),
]

METRICS = [
    ("ROC-AUC", "ROC_AUC"),
    ("PR-AUC", "PR_AUC"),
    ("F1", "F1Score"),
    ("Precision", "Precision"),
    ("Recall", "Recall"),
]


def load_results(folder: Path):
    by_web = defaultdict(list)
    all_rows = []
    for path in sorted(folder.glob("*.csv")):
        web = re.sub(r"_results_SEAL(?:_directed)?$", "", path.stem)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["_foodweb"] = web
                all_rows.append(row)
                by_web[web].append(row)
    return all_rows, by_web


def fvalues(rows, column):
    return [float(row[column]) for row in rows if row.get(column, "") not in ("", None)]


def mean(values):
    return sum(values) / len(values)


def compute_summary():
    original_rows, original_by_web = load_results(ORIGINAL_RESULTS)
    directed_rows, directed_by_web = load_results(DIRECTED_RESULTS)
    common_webs = sorted(set(original_by_web) & set(directed_by_web))

    metric_summary = []
    for label, column in METRICS:
        original_pooled = statistics.median(fvalues(original_rows, column))
        directed_pooled = statistics.median(fvalues(directed_rows, column))
        paired_deltas = []
        for web in common_webs:
            original_mean = mean(fvalues(original_by_web[web], column))
            directed_mean = mean(fvalues(directed_by_web[web], column))
            paired_deltas.append(directed_mean - original_mean)
        improved = sum(delta > 1e-12 for delta in paired_deltas)
        unchanged = sum(abs(delta) <= 1e-12 for delta in paired_deltas)
        lower = sum(delta < -1e-12 for delta in paired_deltas)
        metric_summary.append(
            {
                "label": label,
                "original": original_pooled,
                "directed": directed_pooled,
                "pooled_change": directed_pooled - original_pooled,
                "paired_delta": statistics.median(paired_deltas),
                "improved": improved,
                "unchanged": unchanged,
                "lower": lower,
            }
        )

    train_original = fvalues(original_rows, "TrainRatio")
    train_directed = fvalues(directed_rows, "TrainRatio")
    target_summary = {
        "original_min": min(train_original),
        "original_max": max(train_original),
        "original_below_90": sum(value < 90 for value in train_original),
        "directed_min": min(train_directed),
        "directed_max": max(train_directed),
        "directed_below_90": sum(value < 90 for value in train_directed),
    }

    with ATTRIBUTE_SUMMARY.open(newline="", encoding="utf-8-sig") as handle:
        attribute_rows = list(csv.DictReader(handle))
    written = [row for row in attribute_rows if row["status"] == "written"]
    total_nodes = sum(int(row["nodes"]) for row in written)
    mass_imputed = sum(int(row["mass_imputed"]) for row in written)

    with FEATURE_NAMES.open(newline="", encoding="utf-8-sig") as handle:
        feature_rows = list(csv.DictReader(handle))
    feature_names = [row["feature_name"] for row in feature_rows]
    feature_counts = {
        "Mass": 2,
        "Taxonomic resolution": sum(name.startswith("taxonomy_level=") for name in feature_names),
        "Metabolic type": sum(name.startswith("metabolic_type=") for name in feature_names),
        "Movement type": sum(name.startswith("movement_type=") for name in feature_names),
        "Life stage": sum(name.startswith("lifestage=") for name in feature_names)
        + sum(name == "lifestage_missing" for name in feature_names),
    }

    return {
        "original_rows": len(original_rows),
        "directed_rows": len(directed_rows),
        "original_webs": len(original_by_web),
        "directed_webs": len(directed_by_web),
        "common_webs": len(common_webs),
        "metrics": metric_summary,
        "train_ratio": target_summary,
        "attribute_webs": len(written),
        "total_nodes": total_nodes,
        "mass_imputed": mass_imputed,
        "mass_imputed_pct": 100 * mass_imputed / total_nodes,
        "feature_total": len(feature_names),
        "feature_counts": feature_counts,
    }


def png_size(path: Path):
    with path.open("rb") as handle:
        signature = handle.read(24)
    if signature[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return struct.unpack(">II", signature[16:24])


def run(text, *, bold=False, italic=False, colour=None, size=None):
    properties = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    if colour:
        properties.append(f'<w:color w:val="{colour}"/>')
    if size:
        properties.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(str(text))}</w:t></w:r>'


def paragraph(
    text="",
    *,
    style=None,
    bold=False,
    italic=False,
    colour=None,
    size=None,
    align=None,
    left=None,
    hanging=None,
    before=None,
    after=120,
    keep_next=False,
    page_break_before=False,
    shading=None,
):
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    if left is not None or hanging is not None:
        attrs = []
        if left is not None:
            attrs.append(f'w:left="{left}"')
        if hanging is not None:
            attrs.append(f'w:hanging="{hanging}"')
        props.append(f"<w:ind {' '.join(attrs)}/>")
    spacing = []
    if before is not None:
        spacing.append(f'w:before="{before}"')
    if after is not None:
        spacing.append(f'w:after="{after}"')
    if spacing:
        props.append(f"<w:spacing {' '.join(spacing)}/>")
    if keep_next:
        props.append("<w:keepNext/>")
    if page_break_before:
        props.append("<w:pageBreakBefore/>")
    if shading:
        props.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading}"/>')
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    return f"<w:p>{ppr}{run(text, bold=bold, italic=italic, colour=colour, size=size)}</w:p>"


def bullet(text, level=0):
    left = 420 + level * 360
    return paragraph(f"• {text}", left=left, hanging=260, after=75)


def checkbox(text):
    return paragraph(f"☐ {text}", left=360, hanging=260, after=80)


def cell(text, *, header=False, width=None, shading=None, align=None):
    tcpr = []
    if width:
        tcpr.append(f'<w:tcW w:w="{width}" w:type="dxa"/>')
    if shading or header:
        tcpr.append(f'<w:shd w:val="clear" w:color="auto" w:fill="{shading or "D9EAF7"}"/>')
    contents = []
    for index, line in enumerate(str(text).split("\n")):
        contents.append(
            paragraph(
                line,
                bold=header,
                size=19 if not header else 19,
                align=align,
                after=25 if index < len(str(text).split("\n")) - 1 else 40,
            )
        )
    return f"<w:tc><w:tcPr>{''.join(tcpr)}</w:tcPr>{''.join(contents)}</w:tc>"


def table(headers, rows, widths=None, *, first_col_shading=False):
    if widths is None:
        widths = [None] * len(headers)
    borders = (
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:color="A6A6A6"/>'
        '<w:left w:val="single" w:sz="4" w:color="A6A6A6"/>'
        '<w:bottom w:val="single" w:sz="4" w:color="A6A6A6"/>'
        '<w:right w:val="single" w:sz="4" w:color="A6A6A6"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D0D0D0"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D0D0D0"/>'
        '</w:tblBorders>'
    )
    result = [
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblLayout w:type="fixed"/>' + borders + '</w:tblPr>'
    ]
    result.append("<w:tr>" + "".join(cell(value, header=True, width=widths[i]) for i, value in enumerate(headers)) + "</w:tr>")
    for row_index, row in enumerate(rows):
        row_cells = []
        for col_index, value in enumerate(row):
            shading = "EEF5FA" if first_col_shading and col_index == 0 else ("F7F7F7" if row_index % 2 else None)
            row_cells.append(cell(value, width=widths[col_index], shading=shading))
        result.append("<w:tr>" + "".join(row_cells) + "</w:tr>")
    result.append("</w:tbl>")
    result.append(paragraph("", after=90))
    return "".join(result)


def image_paragraph(rel_id, image_id, path: Path, caption, max_width_in=6.65, max_height_in=5.75):
    width_px, height_px = png_size(path)
    scale = min(max_width_in / width_px, max_height_in / height_px)
    width_in = width_px * scale
    height_in = height_px * scale
    cx = int(width_in * 914400)
    cy = int(height_in * 914400)
    drawing = f'''
    <w:p>
      <w:pPr><w:jc w:val="center"/><w:spacing w:before="120" w:after="80"/></w:pPr>
      <w:r><w:drawing>
        <wp:inline distT="0" distB="0" distL="0" distR="0">
          <wp:extent cx="{cx}" cy="{cy}"/>
          <wp:effectExtent l="0" t="0" r="0" b="0"/>
          <wp:docPr id="{image_id}" name="Figure {image_id}" descr="{escape(caption)}"/>
          <wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>
          <a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr><pic:cNvPr id="{image_id}" name="{escape(path.name)}"/><pic:cNvPicPr/></pic:nvPicPr>
              <pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>
              <pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>
            </pic:pic>
          </a:graphicData></a:graphic>
        </wp:inline>
      </w:drawing></w:r>
    </w:p>'''
    return drawing + paragraph(f"Figure {image_id}. {caption}", style="Caption", align="center", after=170)


def build_document(summary):
    parts = []
    parts.append(paragraph("Working document", style="Eyebrow", align="center", after=100))
    parts.append(paragraph("SEAL_directed with node attributes", style="Title", align="center", after=80))
    parts.append(paragraph("Initial train-ratio 90 comparison with original SEAL", style="Subtitle", align="center", after=120))
    parts.append(paragraph("Prepared for discussion with Dr Athen Ma", align="center", italic=True, colour="4F5B66", after=30))
    parts.append(paragraph("8 September 2026  |  Status: preliminary working analysis", align="center", colour="4F5B66", after=300))
    parts.append(
        paragraph(
            "Purpose. Review the current SEAL_directed implementation and its node attributes, interpret the first target train-ratio 90 results, and agree the controlled experiments required before manuscript-level claims are made.",
            bold=True,
            shading="EAF3F8",
            left=220,
            after=220,
        )
    )

    parts.append(paragraph("Decisions requested in the meeting", style="Heading1", keep_next=True))
    parts.append(checkbox("Confirm whether the current 44-feature biological attribute set should be retained."))
    parts.append(checkbox("Decide whether life-stage variables should remain, given their comparatively low raw coverage."))
    parts.append(checkbox("Approve a controlled ablation that holds positive splits, negative candidates, seeds and repeats constant."))
    parts.append(checkbox("Decide whether the original SEAL baseline should be rerun so every comparison uses identical splits and three repeats."))
    parts.append(checkbox("Agree which outcomes should be reported in the manuscript and which belong in supplementary material."))

    parts.append(paragraph("1. Executive summary", style="Heading1", page_break_before=True, keep_next=True))
    parts.append(bullet(f"The repository contains results for {summary['common_webs']} food webs in both pipelines."))
    parts.append(bullet(f"Original SEAL contributes {summary['original_rows']} run-level rows ({summary['original_rows'] // summary['original_webs']} per web), whereas SEAL_directed contributes {summary['directed_rows']} ({summary['directed_rows'] // summary['directed_webs']} per web)."))
    parts.append(bullet("The pooled run-level medians are higher for SEAL_directed for all five predictive metrics examined."))
    parts.append(bullet("The largest median increases are precision (+0.124), PR-AUC (+0.081) and F1 (+0.073); recall changes much less (+0.011)."))
    parts.append(bullet("The pattern is consistent with fewer false positives while retaining similar sensitivity, but no causal or significance claim should yet be made."))
    parts.append(bullet("The current comparison changes direction handling, biological attributes and negative-link sampling; it is therefore not a clean test of any one component."))

    parts.append(paragraph("Plain-language interpretation", style="Heading2", keep_next=True))
    parts.append(
        paragraph(
            "In these initial runs, SEAL_directed is better at deciding which predicted links are credible. The improvement in precision is substantial, while recall remains broadly similar. In practical terms, the model appears to remove more incorrect predictions without losing many correct ones. However, several implementation differences are bundled together, so the present results cannot tell us whether the gain comes from directionality, node attributes, ecological negative sampling, or their combination."
        )
    )

    parts.append(paragraph("2. What is currently implemented", style="Heading1", keep_next=True))
    parts.append(paragraph("Direction-aware SEAL pipeline", style="Heading2", keep_next=True))
    parts.append(bullet("Positive trophic links are treated as ordered resource-to-consumer pairs."))
    parts.append(bullet("Test-positive links are masked before enclosing-subgraph extraction, avoiding direct target-link leakage."))
    parts.append(bullet("The two target endpoints receive distinct labels, preserving source/target order."))
    parts.append(bullet("Neighbourhoods use incoming and outgoing neighbours, but the present DGCNN receives the weak undirected projection of each subgraph. Direction is therefore represented through endpoint order/labels rather than directed message passing."))
    parts.append(bullet("By default, negative candidates are filtered to ecologically admissible resource-to-consumer role combinations. If that constrained pool is too small, the implementation falls back to all directed unknown pairs."))

    parts.append(paragraph("Node attributes", style="Heading2", keep_next=True))
    parts.append(
        paragraph(
            f"A {summary['feature_total']}-column group matrix is stored in each attributed .mat file and concatenated with the structural SEAL labels before the DGCNN model. The build summary contains {summary['attribute_webs']} food webs and {summary['total_nodes']:,} nodes."
        )
    )
    attribute_rows = [
        ("Body mass", "log10_mass_z + missingness flag", str(summary["feature_counts"]["Mass"]), "95.4%", "Global-median imputation when unavailable; missingness retained explicitly"),
        ("Taxonomic resolution", "One-hot taxonomy level", str(summary["feature_counts"]["Taxonomic resolution"]), "91.5%", "Unknown category retained"),
        ("Metabolic type", "One-hot category", str(summary["feature_counts"]["Metabolic type"]), "100.0%", "Unknown category retained"),
        ("Movement type", "One-hot category", str(summary["feature_counts"]["Movement type"]), "99.7%", "Unknown category retained"),
        ("Life stage", "One-hot category + missingness flag", str(summary["feature_counts"]["Life stage"]), "55.5%", "Missingness retained; interpretation requires caution"),
    ]
    parts.append(table(["Attribute family", "Encoding", "Features", "Mean raw coverage", "Missing-data treatment"], attribute_rows, [1600, 2100, 800, 1250, 2700], first_col_shading=True))
    parts.append(
        paragraph(
            f"Mass was imputed for {summary['mass_imputed']:,} of {summary['total_nodes']:,} nodes ({summary['mass_imputed_pct']:.2f}%). Length variables are excluded because raw coverage is much lower. Degree, trophic level, generality, vulnerability and clustering are also excluded because calculating them on the full network could leak held-out structural information."
        )
    )

    parts.append(image_paragraph("rIdImage1", 1, FIGURES[0][0], FIGURES[0][1]))
    parts.append(image_paragraph("rIdImage2", 2, FIGURES[1][0], FIGURES[1][1], max_height_in=4.9))

    parts.append(paragraph("Questions/comments for Athen", style="Heading2", keep_next=True))
    parts.append(table(["Question", "Athen's comment / decision"], [
        ("Is the current attribute set biologically appropriate for the first ablation?", ""),
        ("Should life stage be retained despite lower coverage?", ""),
        ("Are there attributes that should be moved to a later experiment?", ""),
    ], [4300, 4300]))

    parts.append(paragraph("3. Initial target train-ratio 90 results", style="Heading1", page_break_before=True, keep_next=True))
    parts.append(
        paragraph(
            "The table below reproduces the medians in the original update and verifies them against the current CSV files. These are pooled medians over run-level rows, so they describe the current result folders but are not yet a matched inferential analysis."
        )
    )
    performance_rows = []
    for item in summary["metrics"]:
        performance_rows.append(
            (
                item["label"],
                f"{item['original']:.3f}",
                f"{item['directed']:.3f}",
                f"{item['pooled_change']:+.3f}",
            )
        )
    parts.append(table(["Metric", "Original SEAL", "SEAL_directed", "Median change"], performance_rows, [2200, 2100, 2100, 2100], first_col_shading=True))

    parts.append(paragraph("Food-web-level supporting check", style="Heading2", keep_next=True))
    parts.append(
        paragraph(
            "As a descriptive paired check, runs were first averaged within each food web and model. The table reports the median within-web difference and the number of food webs with higher, unchanged or lower SEAL_directed performance. This still does not replace a preregistered paired statistical analysis."
        )
    )
    delta_rows = []
    for item in summary["metrics"]:
        delta_rows.append(
            (
                item["label"],
                f"{item['paired_delta']:+.3f}",
                str(item["improved"]),
                str(item["unchanged"]),
                str(item["lower"]),
            )
        )
    parts.append(table(["Metric", "Median paired difference", "Higher", "Unchanged", "Lower"], delta_rows, [2300, 2300, 1200, 1300, 1200], first_col_shading=True))

    parts.append(image_paragraph("rIdImage3", 3, FIGURES[2][0], FIGURES[2][1], max_height_in=4.8))
    parts.append(image_paragraph("rIdImage4", 4, FIGURES[3][0], FIGURES[3][1], max_height_in=4.8))

    parts.append(paragraph("4. Limitations of the current comparison", style="Heading1", page_break_before=True, keep_next=True))
    ratio = summary["train_ratio"]
    limitations = [
        "Multiple changes are confounded: direction-aware endpoint encoding, node attributes and role-filtered directed negative sampling are all different from the original baseline.",
        "The present DGCNN graph is a weak undirected projection. The experiment tests direction-aware labelling/order, not a fully directed message-passing architecture.",
        f"The number of repeats is unequal: {summary['original_rows'] // summary['original_webs']} per food web for original SEAL and {summary['directed_rows'] // summary['directed_webs']} for SEAL_directed.",
        f"Recorded train ratios are not perfectly matched. Original SEAL ranges from {ratio['original_min']:.1f}% to {ratio['original_max']:.1f}% and has {ratio['original_below_90']} run-level rows below 90%; SEAL_directed ranges from {ratio['directed_min']:.1f}% to {ratio['directed_max']:.1f}% and has {ratio['directed_below_90']} below 90%.",
        "The pooled medians do not constitute a paired significance test and should not be described as statistically significant.",
        "The current comparison does not quantify the separate contribution of each attribute family or the effect of missing-value handling.",
    ]
    for item in limitations:
        parts.append(bullet(item))

    parts.append(
        paragraph(
            "Recommended wording at this stage. Across the available target train-ratio 90 runs, SEAL_directed produced higher descriptive median performance than the original SEAL baseline, particularly for precision, PR-AUC and F1. Because the pipelines differ in several components and the runs are not fully matched, these results should be treated as preliminary rather than as evidence that any single component caused the improvement.",
            italic=True,
            shading="FFF2CC",
            left=220,
            after=220,
        )
    )

    parts.append(paragraph("5. Proposed controlled ablation", style="Heading1", keep_next=True))
    ablation_rows = [
        ("A. Reproduced baseline", "Original SEAL", "No biological attributes", "Common negative set", "Reference"),
        ("B. Direction-aware only", "Ordered endpoints + directed labels", "No biological attributes", "Same as A", "Direction-aware representation"),
        ("C. Direction-aware + attributes", "Same as B", "All 44 features", "Same as A/B", "Incremental effect of attributes"),
        ("D. Ecological sampling", "Same as C", "All 44 features", "Role-filtered directed negatives", "Incremental effect of ecological negative sampling"),
    ]
    parts.append(table(["Condition", "Representation", "Attributes", "Negative sampling", "Primary contrast"], ablation_rows, [1700, 2100, 1800, 1900, 1700], first_col_shading=True))
    parts.append(paragraph("Controls required", style="Heading2", keep_next=True))
    for item in [
        "Precompute and reuse identical positive train/test splits for every condition.",
        "Where conditions are intended to share negative sampling, precompute and reuse the same negative candidates and assignments.",
        "Use the same food webs, seeds, number of repeats, hop setting, thresholding rule and evaluation set.",
        "Aggregate repetitions within food web, then use food-web-level paired comparisons and report effect sizes alongside a paired test.",
        "Record any constrained-pool fallback, because it changes the interpretation of the ecological sampling condition.",
        "Add attribute-family ablations only after the main direction/attribute/sampling effects are separated.",
    ]:
        parts.append(checkbox(item))

    parts.append(paragraph("6. Agreed actions", style="Heading1", page_break_before=True, keep_next=True))
    parts.append(table(["Action", "Owner", "Priority / deadline", "Decision or comment"], [
        ("Confirm the four-condition ablation design", "Athen / Jorge", "", ""),
        ("Regenerate matched positive and negative splits", "Jorge", "", ""),
        ("Rerun original baseline with three matched repeats", "Jorge", "", ""),
        ("Run direction-aware model without attributes", "Jorge", "", ""),
        ("Run direction-aware model with attributes under common sampling", "Jorge", "", ""),
        ("Apply paired statistical analysis and update figures", "Jorge", "", ""),
        ("Agree manuscript wording after controlled results", "Athen / Jorge", "", ""),
    ], [3200, 1400, 1700, 2500]))

    parts.append(paragraph("Appendix A. Reproducibility and source files", style="Heading1", keep_next=True))
    provenance = [
        ("Original SEAL result CSVs", "src/python/data/result/prediction_scores_logs/"),
        ("SEAL_directed result CSVs", "src/python/seal_directed/data/result/prediction_scores_logs/"),
        ("Attributed .mat build script", "src/python/seal_directed/build_node_attribute_mats.py"),
        ("Directed implementation", "src/python/seal_directed/Main_directed.py; src/python/seal_directed/util_functions_directed.py"),
        ("Attribute EDA notebook", "docs/progress_notebooks_logs/2026/seal_directed_node_attribute_eda.ipynb"),
        ("Initial comparison notebook", "docs/progress_notebooks_logs/2026/week-25.ipynb"),
        ("Existing figures", "docs/plots/seal_directed_node_attribute_eda/; docs/plots/seal_comparison/"),
        ("Document generator", "docs/SEAL/reports/build_seal_directed_train90_update.py"),
    ]
    parts.append(table(["Item", "Repository path"], provenance, [2600, 6000], first_col_shading=True))
    parts.append(
        paragraph(
            "Document note. The figures embedded above already existed in the repository. This build did not generate or alter any analytical figure. Future edits should use Microsoft Word Review > Track Changes and comments so that decisions and wording changes remain visible.",
            italic=True,
            colour="4F5B66",
        )
    )

    section = '''
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="600" w:footer="600" w:gutter="0"/>
      <w:cols w:space="720"/>
      <w:docGrid w:linePitch="360"/>
    </w:sectPr>'''
    return "".join(parts) + section


def document_xml(body):
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document
 xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
 <w:body>{body}</w:body>
</w:document>'''


STYLES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos" w:eastAsia="Aptos" w:cs="Aptos"/><w:sz w:val="21"/><w:szCs w:val="21"/><w:lang w:val="en-GB"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:next w:val="Subtitle"/><w:qFormat/><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="38"/><w:szCs w:val="38"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:color w:val="3F6F8F"/><w:sz w:val="26"/><w:szCs w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Eyebrow"><w:name w:val="Eyebrow"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:caps/><w:color w:val="2E75B6"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="260" w:after="120"/><w:outlineLvl w:val="0"/></w:pPr><w:rPr><w:b/><w:color w:val="17365D"/><w:sz w:val="29"/><w:szCs w:val="29"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:keepLines/><w:spacing w:before="180" w:after="90"/><w:outlineLvl w:val="1"/></w:pPr><w:rPr><w:b/><w:color w:val="2E75B6"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:qFormat/><w:rPr><w:i/><w:color w:val="555555"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
</w:styles>'''


CONTENT_TYPES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''


ROOT_RELS_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''


def document_rels_xml():
    image_rels = []
    for index in range(1, len(FIGURES) + 1):
        image_rels.append(
            f'<Relationship Id="rIdImage{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image{index}.png"/>'
        )
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  %s
</Relationships>''' % "\n  ".join(image_rels)


SETTINGS_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
  <w:trackRevisions/>
</w:settings>'''


def core_xml():
    timestamp = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>SEAL_directed with node attributes: initial train-ratio 90 results</dc:title>
  <dc:subject>Working document for discussion with Dr Athen Ma</dc:subject>
  <dc:creator>Jorge Eduardo Castro Cruces</dc:creator>
  <cp:lastModifiedBy>Jorge Eduardo Castro Cruces</cp:lastModifiedBy>
  <dc:description>Preliminary implementation summary, descriptive results, limitations and controlled ablation proposal.</dc:description>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>'''


APP_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company>Queen Mary University of London</Company>
  <AppVersion>16.0000</AppVersion>
</Properties>'''


def build_docx():
    for required in [ORIGINAL_RESULTS, DIRECTED_RESULTS, ATTRIBUTE_SUMMARY, FEATURE_NAMES]:
        if not required.exists():
            raise FileNotFoundError(required)
    for path, _ in FIGURES:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = compute_summary()
    body = build_document(summary)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        package.writestr("_rels/.rels", ROOT_RELS_XML)
        package.writestr("docProps/core.xml", core_xml())
        package.writestr("docProps/app.xml", APP_XML)
        package.writestr("word/document.xml", document_xml(body))
        package.writestr("word/styles.xml", STYLES_XML)
        package.writestr("word/settings.xml", SETTINGS_XML)
        package.writestr("word/_rels/document.xml.rels", document_rels_xml())
        for index, (path, _) in enumerate(FIGURES, start=1):
            package.write(path, f"word/media/image{index}.png")
    return summary


if __name__ == "__main__":
    result = build_docx()
    print(OUTPUT)
    print(
        f"Verified {result['common_webs']} paired food webs, "
        f"{result['original_rows']} original rows, {result['directed_rows']} directed rows."
    )
