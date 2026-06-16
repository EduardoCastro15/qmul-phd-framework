import csv
import json
from pathlib import Path


EDA_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = EDA_DIR / "tables"
NOTEBOOK_PATH = EDA_DIR / "DAPSTOM_EDA.ipynb"


def read_rows(name):
    with (TABLE_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(value):
    if value in (None, ""):
        return 0
    return int(float(value))


def as_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


def fmt_int(value):
    return f"{as_int(value):,}"


def fmt_float(value, digits=1):
    return f"{as_float(value):,.{digits}f}"


def pct(numer, denom, digits=1):
    denom = as_float(denom)
    if denom == 0:
        return "n/a"
    return f"{100 * as_float(numer) / denom:.{digits}f}%"


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def top_rows(name, limit):
    return read_rows(name)[:limit]


def get_count(table_name):
    for row in read_rows("table_row_counts"):
        if row["table_name"] == table_name:
            return as_int(row["row_count"])
    return 0


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def build():
    hauls = get_count("HAUL 6-4 COMBINED")
    predators = get_count("PREDATOR 6-4 COMBINED")
    prey = get_count("PREY 6-4 COMBINED")
    provenance = get_count("PROVENANCE 6-4 COMBINED")
    murray = get_count("MURRAY_TAXONOMY")

    years = [as_int(r["year"]) for r in read_rows("haul_temporal_by_year") if r["year"]]
    min_year, max_year = min(years), max(years)

    spatial = read_rows("haul_spatial_resolution")
    spatial_rows = [
        [r["resolution"].strip(), fmt_int(r["haul_rows"]), pct(r["haul_rows"], r["total_hauls"])]
        for r in spatial
    ]

    pool = read_rows("predator_pooling_summary")
    total_pred_rows = sum(as_int(r["predator_rows"]) for r in pool)
    total_stomachs = sum(as_float(r["total_stomachs"]) for r in pool)
    pool_rows = [
        [
            r["pooled"] or "(missing)",
            fmt_int(r["predator_rows"]),
            pct(r["predator_rows"], total_pred_rows),
            fmt_int(r["total_stomachs"]),
            pct(r["total_stomachs"], total_stomachs),
            fmt_float(r["avg_num_stomachs"], 2),
        ]
        for r in pool
    ]

    evidence = read_rows("prey_evidence_stats")[0]
    taxonomy = read_rows("taxonomy_coverage")[0]
    edge_total = as_int(read_rows("global_unique_edge_pairs")[0]["unique_predator_prey_pairs"])
    edge_positive = as_int(read_rows("global_unique_edge_pairs_positive_prey_tsn")[0][
        "unique_predator_prey_pairs_positive_prey_tsn"
    ])

    sea_rows = []
    for r in top_rows("haul_spatial_by_sea", 10):
        sea_rows.append([
            r["sea"],
            fmt_int(r["haul_rows"]),
            pct(r["with_lat_lon"], r["haul_rows"]),
            pct(r["with_ices_rect"], r["haul_rows"]),
            r["distinct_ices_rectangles"],
        ])

    missing_rows = []
    for r in read_rows("critical_field_missingness"):
        nulls = as_int(r["null_rows"])
        blanks = as_int(r["blank_text_rows"])
        if nulls or blanks:
            missing_rows.append([
                r["table_name"],
                r["field_name"],
                fmt_int(r["total_rows"]),
                fmt_int(nulls),
                fmt_int(blanks),
            ])

    top_predator_rows = [
        [r["predator_name"], fmt_int(r["predator_rows"]), fmt_int(r["total_stomachs"]), fmt_float(r["avg_mean_length_cm"], 1)]
        for r in top_rows("top_predators", 10)
    ]

    top_prey_rows = [
        [r["prey_name"], r["prey_tsn"], fmt_int(r["prey_rows"]), fmt_int(r["total_min_num"]), fmt_float(r["total_cpw"], 1)]
        for r in top_rows("top_prey", 10)
    ]

    negative_rows = [
        [r["prey_tsn"], r["prey_name"], fmt_int(r["prey_rows"]), fmt_int(r["total_min_num"])]
        for r in top_rows("negative_tsn_prey_categories", 10)
    ]

    edge_rows = [
        [
            r["predator_name"],
            r["prey_name"],
            fmt_int(r["prey_records"]),
            fmt_int(r["total_min_num"]),
            fmt_float(r["total_cpw"], 1),
        ]
        for r in top_rows("top_predator_prey_edges", 12)
    ]

    potential_rows = []
    for r in top_rows("edge_pairs_by_sea_decade", 12):
        sea = r.get("sea") or r.get("SEA")
        decade = r.get("decade") or r.get("DECADE")
        potential_rows.append([
            sea,
            decade,
            fmt_int(r["unique_predator_prey_pairs"]),
            fmt_int(r["prey_records"]),
        ])

    qualifier_rows = [
        [r["qual_code"], r["qual_description"], fmt_int(r["prey_rows"]), fmt_int(r["total_min_num"])]
        for r in top_rows("prey_qualifiers", 10)
    ]

    table_count_rows = [
        [r["table_name"], fmt_int(r["row_count"])]
        for r in read_rows("table_row_counts")
    ]

    cells = [
        markdown_cell(
            "# DAPSTOM 6.4 Access Database - Initial EDA\n\n"
            "This notebook summarises the Microsoft Access copy received from Cefas. "
            "It is intentionally conservative: the source `.accdb` is not overwritten, and the analysis uses "
            "derived CSV summaries under `tables/` rather than a full database export.\n\n"
            "**Core question:** is the database suitable for constructing predator-prey interaction datasets "
            "and later directed food-web/link-prediction experiments?"
        ),
        markdown_cell(
            "## Executive Summary\n\n"
            f"- Core relational chain: `HAUL -> PREDATOR -> PREY`.\n"
            f"- Scale: **{hauls:,} hauls**, **{predators:,} predator records**, and **{prey:,} prey records**.\n"
            f"- Coverage spans **{min_year}-{max_year}** in the haul table.\n"
            f"- Spatial resolution is mixed: {spatial_rows[0][2]} of hauls have point latitude/longitude, "
            f"{spatial_rows[1][2]} have ICES rectangle, and all hauls have ICES division/sea.\n"
            f"- There are **{edge_total:,}** observed predator-prey name pairs, or **{edge_positive:,}** after "
            "removing prey records with negative TSN codes.\n"
            f"- `min_num` and `cpw` are almost complete in the prey table, but `pooled` and `num_stomachs` "
            "must be handled explicitly because pooled records represent many stomachs.\n"
            "- The dataset is viable for food-web construction, but empty/unidentified/digested categories "
            "should be separated from ordinary trophic edges."
        ),
        markdown_cell(
            "## Reproducibility\n\n"
            "The summaries were produced from the working Access copy:\n\n"
            "`data/dapstom_6_4_combined_working_copy.accdb`\n\n"
            "The extractor uses the DBeaver-downloaded `io.github.spannm:ucanaccess:5.1.5` JDBC driver. "
            "To regenerate the CSV summaries from the repo root:\n\n"
            "```bash\n"
            "bash data/processed/dapstom_eda/tools/run_extractor.sh\n"
            "python3 data/processed/dapstom_eda/tools/build_notebook.py\n"
            "```\n\n"
            "Known non-blocking warning: UCanAccess cannot load the saved Access view `Query1` because the view "
            "contains duplicate output column names (`tsn` from predator and prey). This EDA uses base tables "
            "and explicit aliases, so the warning is not material."
        ),
        code_cell(
            "from pathlib import Path\n"
            "import csv, json\n\n"
            "TABLE_DIR = Path('tables')\n"
            "if not TABLE_DIR.exists():\n"
            "    TABLE_DIR = Path('data/processed/dapstom_eda/tables')\n\n"
            "def load_csv(name):\n"
            "    with (TABLE_DIR / f'{name}.csv').open(newline='', encoding='utf-8') as f:\n"
            "        return list(csv.DictReader(f))\n\n"
            "sorted(p.name for p in TABLE_DIR.glob('*.csv'))"
        ),
        markdown_cell(
            "## 1. Table Inventory\n\n"
            "The Access file exposes ten user tables. The core analysis tables are `HAUL`, `PREDATOR`, "
            "`PREY`, `PROVENANCE`, `QUALIFYER`, and `MURRAY_TAXONOMY`.\n\n"
            + md_table(["Table", "Rows"], table_count_rows)
        ),
        markdown_cell(
            "## 2. Spatial and Temporal Coverage\n\n"
            "Spatial resolution varies by record. This matters because later networks may need to be built at "
            "different spatial grains depending on the coverage required.\n\n"
            + md_table(["Resolution", "Hauls", "Coverage"], spatial_rows)
            + "\n\nTop sea-level groups by haul count:\n\n"
            + md_table(["Sea", "Hauls", "With lat/lon", "With ICES rect", "Distinct ICES rects"], sea_rows)
        ),
        markdown_cell(
            "## 3. Pooled vs Individual Stomachs\n\n"
            "The `PREDATOR` table has both individual and pooled records. For modelling, predator rows should "
            "not be interpreted as equal sampling units without considering `pooled` and `num_stomachs`.\n\n"
            + md_table(
                ["Pooled", "Predator rows", "% predator rows", "Total stomachs", "% stomachs", "Avg num_stomachs"],
                pool_rows,
            )
        ),
        markdown_cell(
            "## 4. Prey Evidence and Taxonomy\n\n"
            f"`PREY.min_num` is present for **{fmt_int(evidence['with_min_num'])}/{fmt_int(evidence['prey_rows'])}** "
            f"prey records and sums to **{fmt_int(evidence['total_min_num'])}**. "
            f"`PREY.cpw` is present for **{fmt_int(evidence['with_cpw'])}/{fmt_int(evidence['prey_rows'])}** records.\n\n"
            f"All prey rows have a TSN and match `MURRAY_TAXONOMY` by TSN in this extract. "
            f"APHIA IDs are present for **{fmt_int(taxonomy['with_aphiaid'])}/{fmt_int(taxonomy['prey_rows'])}** "
            f"prey rows ({pct(taxonomy['with_aphiaid'], taxonomy['prey_rows'])}).\n\n"
            "Most prey records have qualifier `Q1` (`NONE`), but life-stage and sex qualifiers are common enough "
            "to preserve during cleaning.\n\n"
            + md_table(["Qualifier", "Description", "Prey rows", "Total min_num"], qualifier_rows)
        ),
        markdown_cell(
            "## 5. Missingness in Critical Fields\n\n"
            "Core IDs and predator/prey names are complete in the inspected fields. Main gaps are finer time fields, "
            "point coordinates, and ICES rectangle blanks.\n\n"
            + md_table(["Table", "Field", "Rows", "Null rows", "Blank text rows"], missing_rows)
        ),
        markdown_cell(
            "## 6. Dominant Predators and Prey\n\n"
            "Predator codes/names are short labels in the Access table. These should be reconciled with taxonomy "
            "before manuscript-quality figures.\n\n"
            "Top predator records:\n\n"
            + md_table(["Predator", "Rows", "Total stomachs", "Avg mean length cm"], top_predator_rows)
            + "\n\nTop prey records:\n\n"
            + md_table(["Prey", "Prey TSN", "Rows", "Total min_num", "Total cpw"], top_prey_rows)
        ),
        markdown_cell(
            "## 7. Non-trophic or Ambiguous Prey Categories\n\n"
            "Negative prey TSN values encode categories such as empty stomachs, digested remains, unknown material, "
            "or other non-standard entries. These are valuable for sampling/absence information but should usually "
            "be filtered or modelled separately from trophic edges.\n\n"
            + md_table(["Prey TSN", "Prey name", "Rows", "Total min_num"], negative_rows)
        ),
        markdown_cell(
            "## 8. Predator-Prey Edge Potential\n\n"
            f"The raw joined data imply **{edge_total:,}** unique predator-prey name pairs. After retaining only "
            f"positive prey TSN values, the potential edge set is **{edge_positive:,}** pairs. This is promising "
            "for link prediction, but edge definitions should be stratified by space/time/source and should not "
            "mix pooled and individual evidence without weighting or uncertainty handling.\n\n"
            "Most frequent observed predator-prey rows:\n\n"
            + md_table(["Predator", "Prey", "Rows", "Total min_num", "Total cpw"], edge_rows)
        ),
        markdown_cell(
            "## 9. Candidate Strata for Food-web Construction\n\n"
            "A practical first pass is to construct networks by `sea x decade`, then later test ICES division or "
            "ICES rectangle strata where sample sizes are sufficient. The table below ranks sea-decade strata by "
            "unique predator-prey pairs.\n\n"
            + md_table(["Sea", "Decade", "Unique pairs", "Prey records"], potential_rows)
        ),
        markdown_cell(
            "## 10. Recommendations for the Next Analysis Step\n\n"
            "1. Preserve the relational export as base tables rather than creating one flat master CSV.\n"
            "2. Build a derived edge table with explicit columns: haul context, predator ID/name/TSN, prey name/TSN, "
            "`min_num`, `cpw`, `pooled`, `num_stomachs`, `num_empty`, provenance and spatial grain.\n"
            "3. Treat `Empty`, negative TSN categories, unknown/digested remains, and broad categories separately.\n"
            "4. For ML link prediction, define positives from observed predator-prey pairs after filtering, and "
            "sample negatives within ecologically comparable strata to avoid trivial absences.\n"
            "5. Use grouped evaluation splits by sea/decade/source/predator where possible; random edge splits "
            "will likely overestimate generalisation.\n"
            "6. Discuss interpretation with Cefas before final modelling choices, especially around pooled records "
            "and geographic resolution."
        ),
        markdown_cell(
            "## Generated Files\n\n"
            "- `tables/table_inventory.csv` and `tables/column_inventory.csv`: schema overview.\n"
            "- `tables/table_row_counts.csv`: row counts for all user tables.\n"
            "- `tables/*summary*.csv`, `tables/top_*.csv`, and `tables/*coverage*.csv`: EDA summaries.\n"
            "- `tools/DapstomEdaExtractor.java`: JDBC extractor used to create the summaries.\n"
            "- `tools/run_extractor.sh`: convenience wrapper for recompilation and extraction."
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
