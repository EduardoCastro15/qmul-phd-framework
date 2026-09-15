#!/usr/bin/env python3
"""Export two compact Excel workbooks for the Figure 5 Wilcoxon results.

Both workbooks are recalculated from the same post-Tukey food-web pairs.
Only unadjusted Wilcoxon p-values are exported. The first workbook contains
one test per ecosystem and metric; the second contains one test per metric
across food webs.
"""

from __future__ import annotations

import datetime as dt
import math
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = (
    REPO_ROOT
    / "src/matlab/data"
    / (
        "result_wlnm_dir_neg_sweep_train_ratios_"
        "10-90_pseudo_properties_Apocrita_neg_const"
    )
    / "retention_protocol"
    / "tukey_iqr_1p5_min50pct_threshold0p50_v1"
    / "statistical_tests_Wilcoxon signed-rank test"
)
PAIR_INPUT = RESULT_DIR / (
    "figure_5_wilcoxon_train60_roleonly_5constraints_"
    "checkconnfalse_adaptivefalse_threshold0p50_after_tukey_pairs.csv"
)
ECOSYSTEM_OUTPUT = RESULT_DIR / "figure_5_wilcoxon_by_ecosystem_train60.xlsx"
FOODWEB_OUTPUT = RESULT_DIR / "figure_5_wilcoxon_by_foodweb_train60.xlsx"

METRIC_ORDER = [
    "Connectance",
    "MeanTrophicHeight",
    "MeanGenerality",
    "MeanVulnerability",
]
METRIC_LABELS = {
    "Connectance": "Connectance",
    "MeanTrophicHeight": "Mean trophic height",
    "MeanGenerality": "Mean generality",
    "MeanVulnerability": "Mean vulnerability",
}
ECOSYSTEM_ORDER = [
    "lakes",
    "streams",
    "marine",
    "terrestrial aboveground",
    "terrestrial belowground",
]
ECOSYSTEM_LABELS = {
    "lakes": "Lakes",
    "streams": "Streams",
    "marine": "Marine",
    "terrestrial aboveground": "Terrestrial aboveground",
    "terrestrial belowground": "Terrestrial belowground",
}


def signed_rank_summary(group: pd.DataFrame) -> dict[str, float | int]:
    differences = np.round(
        group["MeanPseudoAfterFiltering"].to_numpy(dtype=float)
        - group["EmpiricalValue"].to_numpy(dtype=float),
        decimals=12,
    )
    nonzero = differences[differences != 0.0]
    if nonzero.size:
        test = wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        ranks = rankdata(np.abs(nonzero), method="average")
        positive_rank_sum = float(ranks[nonzero > 0.0].sum())
        negative_rank_sum = float(ranks[nonzero < 0.0].sum())
        rank_biserial = (
            (positive_rank_sum - negative_rank_sum)
            / (positive_rank_sum + negative_rank_sum)
        )
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    else:
        statistic = 0.0
        p_value = 1.0
        rank_biserial = 0.0

    return {
        "Number of pairs": int(len(group)),
        "Median empirical": float(group["EmpiricalValue"].median()),
        "Median pseudo after Tukey": float(
            group["MeanPseudoAfterFiltering"].median()
        ),
        "Median difference (pseudo - empirical)": float(np.median(differences)),
        "Wilcoxon statistic": statistic,
        "p-value": p_value,
        "Rank-biserial correlation": rank_biserial,
    }


def load_pairs() -> pd.DataFrame:
    if not PAIR_INPUT.is_file():
        raise FileNotFoundError(PAIR_INPUT)
    pairs = pd.read_csv(PAIR_INPUT)
    required = {
        "FoodWeb",
        "FW_KEY",
        "Ecosystem",
        "Metric",
        "EmpiricalValue",
        "MeanPseudoAfterFiltering",
    }
    missing = sorted(required.difference(pairs.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    pairs["Ecosystem"] = (
        pairs["Ecosystem"].astype(str).str.strip().str.casefold()
    )
    pairs["Metric"] = pairs["Metric"].astype(str).str.strip()
    for column in ["EmpiricalValue", "MeanPseudoAfterFiltering"]:
        pairs[column] = pd.to_numeric(pairs[column], errors="coerce")
    if pairs[["EmpiricalValue", "MeanPseudoAfterFiltering"]].isna().any().any():
        raise ValueError("The paired input contains missing numerical values.")
    if pairs.duplicated(["FW_KEY", "Metric"]).any():
        raise ValueError("Duplicated food-web and metric pairs were found.")
    return pairs


def calculate_ecosystem_results(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ecosystem in ECOSYSTEM_ORDER:
        for metric in METRIC_ORDER:
            group = pairs.loc[
                pairs["Ecosystem"].eq(ecosystem)
                & pairs["Metric"].eq(metric)
            ]
            if group.empty:
                raise ValueError(
                    f"No pairs for ecosystem={ecosystem}, metric={metric}"
                )
            rows.append(
                {
                    "Ecosystem": ECOSYSTEM_LABELS[ecosystem],
                    "Metric": METRIC_LABELS[metric],
                    **signed_rank_summary(group),
                }
            )
    return pd.DataFrame(rows)


def calculate_foodweb_results(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRIC_ORDER:
        group = pairs.loc[pairs["Metric"].eq(metric)]
        if group.empty:
            raise ValueError(f"No pairs for metric={metric}")
        rows.append(
            {
                "Metric": METRIC_LABELS[metric],
                **signed_rank_summary(group),
            }
        )
    return pd.DataFrame(rows)


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def text_cell(ref: str, value: object, style: int = 0) -> str:
    return (
        f'<c r="{ref}" t="inlineStr" s="{style}">'
        f'<is><t>{escape(str(value))}</t></is></c>'
    )


def number_cell(ref: str, value: object, style: int) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return f'<c r="{ref}" s="{style}"/>'
    return f'<c r="{ref}" s="{style}"><v>{float(value):.17g}</v></c>'


def worksheet_xml(frame: pd.DataFrame, widths: list[float]) -> str:
    rows = []
    header_cells = [
        text_cell(f"{column_name(index)}1", column, style=1)
        for index, column in enumerate(frame.columns, start=1)
    ]
    rows.append(f'<row r="1" ht="30" customHeight="1">{"".join(header_cells)}</row>')

    integer_columns = {"Number of pairs"}
    pvalue_columns = {"p-value"}
    for row_number, values in enumerate(frame.itertuples(index=False, name=None), start=2):
        cells = []
        for column_index, (column, value) in enumerate(
            zip(frame.columns, values), start=1
        ):
            ref = f"{column_name(column_index)}{row_number}"
            if isinstance(value, str):
                cells.append(text_cell(ref, value, style=0))
            elif column in integer_columns:
                cells.append(number_cell(ref, value, style=2))
            elif column in pvalue_columns:
                cells.append(number_cell(ref, value, style=4))
            else:
                cells.append(number_cell(ref, value, style=3))
        rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    last_column = column_name(len(frame.columns))
    last_row = len(frame) + 1
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="18"/>
  <cols>{columns}</cols>
  <sheetData>{''.join(rows)}</sheetData>
  <autoFilter ref="A1:{last_column}{last_row}"/>
  <pageMargins left="0.25" right="0.25" top="0.5" bottom="0.5" header="0.2" footer="0.2"/>
  <pageSetup orientation="landscape" fitToWidth="1" fitToHeight="0"/>
</worksheet>'''


STYLES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="2">
    <numFmt numFmtId="164" formatCode="0.000000"/>
    <numFmt numFmtId="165" formatCode="0.000E+00"/>
  </numFmts>
  <fonts count="2">
    <font><sz val="11"/><name val="Aptos"/><family val="2"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Aptos Display"/><family val="2"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E2F3"/></left><right style="thin"><color rgb="FFD9E2F3"/></right><top style="thin"><color rgb="FFD9E2F3"/></top><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="5">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1"><alignment vertical="center"/></xf>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1"><alignment horizontal="center" vertical="center" wrapText="1"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="center" vertical="center"/></xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1" applyAlignment="1"><alignment horizontal="right" vertical="center"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''


CONTENT_TYPES_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

ROOT_RELS_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>'''

WORKBOOK_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView xWindow="0" yWindow="0" windowWidth="24000" windowHeight="12000"/></bookViews>
  <sheets><sheet name="Results" sheetId="1" r:id="rId1"/></sheets>
  <calcPr calcId="191029"/>
</workbook>'''

WORKBOOK_RELS_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

APP_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel</Application>
  <Company>Queen Mary University of London</Company>
</Properties>'''


def core_xml(title: str) -> str:
    timestamp = (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>{escape(title)}</dc:title>
  <dc:creator>Jorge Eduardo Castro Cruces</dc:creator>
  <cp:lastModifiedBy>Jorge Eduardo Castro Cruces</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>'''


def write_xlsx(path: Path, frame: pd.DataFrame, widths: list[float], title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", CONTENT_TYPES_XML)
        package.writestr("_rels/.rels", ROOT_RELS_XML)
        package.writestr("docProps/core.xml", core_xml(title))
        package.writestr("docProps/app.xml", APP_XML)
        package.writestr("xl/workbook.xml", WORKBOOK_XML)
        package.writestr("xl/_rels/workbook.xml.rels", WORKBOOK_RELS_XML)
        package.writestr("xl/styles.xml", STYLES_XML)
        package.writestr("xl/worksheets/sheet1.xml", worksheet_xml(frame, widths))


def main():
    pairs = load_pairs()
    ecosystem_results = calculate_ecosystem_results(pairs)
    foodweb_results = calculate_foodweb_results(pairs)

    expected_counts = {
        "Connectance": 290,
        "Mean trophic height": 288,
        "Mean generality": 290,
        "Mean vulnerability": 290,
    }
    observed_counts = dict(
        zip(foodweb_results["Metric"], foodweb_results["Number of pairs"])
    )
    if observed_counts != expected_counts:
        raise ValueError(
            f"Unexpected food-web pair counts: {observed_counts}"
        )
    if len(ecosystem_results) != 20 or len(foodweb_results) != 4:
        raise ValueError("Unexpected number of output tests.")

    ecosystem_widths = [27, 23, 16, 19, 25, 32, 19, 15, 26]
    foodweb_widths = [23, 16, 19, 25, 32, 19, 15, 26]
    write_xlsx(
        ECOSYSTEM_OUTPUT,
        ecosystem_results,
        ecosystem_widths,
        "Figure 5 Wilcoxon signed-rank tests by ecosystem type",
    )
    write_xlsx(
        FOODWEB_OUTPUT,
        foodweb_results,
        foodweb_widths,
        "Figure 5 Wilcoxon signed-rank tests by food web",
    )
    print(ECOSYSTEM_OUTPUT)
    print(FOODWEB_OUTPUT)
    print("Ecosystem tests: 20")
    print("Food-web tests: 4")
    print("Exported p-values: unadjusted two-sided Wilcoxon p-values")


if __name__ == "__main__":
    main()
