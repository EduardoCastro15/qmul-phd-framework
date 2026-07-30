import csv
import json
from pathlib import Path

from derive_network_metrics import derive


EDA_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = EDA_DIR / "tables"
NOTEBOOK_PATH = EDA_DIR / "DAPSTOM_EDA.ipynb"


def read_rows(name):
    with (TABLE_DIR / f"{name}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_int(value):
    if value in (None, ""):
        return 0
    return int(round(float(value)))


def as_float(value):
    if value in (None, ""):
        return 0.0
    return float(value)


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
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def get_count(table_name):
    for row in read_rows("table_row_counts"):
        if row["table_name"] == table_name:
            return as_int(row["row_count"])
    return 0


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


def build():
    networks, network_metrics, representatives = derive()

    manifest = json.loads((TABLE_DIR / "eda_manifest.json").read_text(encoding="utf-8"))
    hauls = get_count("HAUL 6-4 COMBINED")
    predator_rows = get_count("PREDATOR 6-4 COMBINED")
    prey_rows = get_count("PREY 6-4 COMBINED")
    provenance_rows = get_count("PROVENANCE 6-4 COMBINED")
    years = [as_int(row["year"]) for row in read_rows("haul_temporal_by_year") if row["year"]]
    min_year, max_year = min(years), max(years)

    raw_network = read_rows("global_network_summary")[0]
    positive_network = read_rows("global_network_summary_positive_prey_tsn")[0]
    raw_name_edges = as_int(
        read_rows("global_unique_edge_pairs")[0]["unique_predator_prey_pairs"]
    )
    positive_name_edges = as_int(
        read_rows("global_unique_edge_pairs_positive_prey_tsn")[0][
            "unique_predator_prey_pairs_positive_prey_tsn"
        ]
    )
    excluded_prey_rows = as_int(raw_network["prey_records"]) - as_int(
        positive_network["prey_records"]
    )

    global_tsn_edges = {
        (edge["predator_tsn"], edge["prey_tsn"])
        for edges in networks.values()
        for edge in edges
    }
    global_nodes = {node for edge in global_tsn_edges for node in edge}
    global_self_loops = sum(predator == prey for predator, prey in global_tsn_edges)
    global_s = len(global_nodes)
    global_l = len(global_tsn_edges)
    global_density = (
        (global_l - global_self_loops) / (global_s * (global_s - 1))
        if global_s > 1
        else 0.0
    )
    global_connectance = global_l / (global_s * global_s) if global_s else 0.0

    spatial_rows = [
        [
            row["resolution"].strip(),
            fmt_int(row["haul_rows"]),
            pct(row["haul_rows"], row["total_hauls"]),
        ]
        for row in read_rows("haul_spatial_resolution")
    ]
    sea_rows = [
        [
            row["sea"],
            fmt_int(row["haul_rows"]),
            pct(row["with_lat_lon"], row["haul_rows"]),
            pct(row["with_ices_rect"], row["haul_rows"]),
            fmt_int(row["distinct_ices_rectangles"]),
        ]
        for row in read_rows("haul_spatial_by_sea")[:10]
    ]

    pool = read_rows("predator_pooling_summary")
    total_stomachs = sum(as_float(row["total_stomachs"]) for row in pool)
    pool_rows = [
        [
            row["pooled"] or "(missing)",
            fmt_int(row["predator_rows"]),
            pct(row["predator_rows"], predator_rows),
            fmt_int(row["total_stomachs"]),
            pct(row["total_stomachs"], total_stomachs),
            fmt_num(row["avg_num_stomachs"]),
        ]
        for row in pool
    ]

    provenance_summary = read_rows("provenance_source_summary")
    provenance_table = [
        [
            row["source_type"] or "(missing)",
            fmt_int(row["provenance_rows"]),
            fmt_int(row["linked_hauls"]),
        ]
        for row in provenance_summary
    ]

    variable_rows = [
        ["PROVENANCE", "`cruise_name`", "Source/campaign identifier; FK from HAUL", "identifier", "Links every haul to source documentation."],
        ["PROVENANCE", "`uploaded`, `source_type`, `data_input`, `data_derived_from`", "Version introduced, source class, digitiser and source description", "categorical/text", "All 741 provenance rows are populated."],
        ["HAUL", "`haul_id`", "Sampling-event identifier; PK", "identifier", "Parent key for predator records."],
        ["HAUL", "`Year`, `Month`, `Day`, `date`", "Sampling date", "calendar fields", "Month/day/date are less complete than year."],
        ["HAUL", "`shot_lat_dd`, `shot_lon_dd`", "Sampling position", "decimal degrees", "Point coordinates are partial."],
        ["HAUL", "`ices_rect`, `ices_division`, `sea`", "Nested spatial descriptors", "categorical", "ICES rectangle is 0.5° latitude × 1° longitude."],
        ["HAUL", "`shot_depth_m`", "Sampling depth", "m", "Stored as text; parseability and values such as `INTERTIDAL` require handling."],
        ["PREDATOR", "`pred_id`, `haul_id`", "Predator-record PK and FK to HAUL", "identifier", "A record may represent one or pooled stomachs."],
        ["PREDATOR", "`pred`, `tsn`", "Local predator label and ITIS Taxonomic Serial Number", "code/identifier", "Use TSN, not the short label, as the graph node key."],
        ["PREDATOR", "`pred_length_cm`, `mean_length_cm`", "Observed or pooled mean predator length", "cm", "Important for ontogenetic diet effects."],
        ["PREDATOR", "`pred_wgt_g`", "Predator weight", "g", "Stored as text and incompletely populated."],
        ["PREDATOR", "`pooled`, `num_stomachs`, `num_empty`", "Sampling-unit flag and represented stomach quantities", "flag/count-like", "Some recorded quantities are fractional; do not force integer type blindly."],
        ["PREY", "`id`, `pred_id`", "Prey-record PK and FK to PREDATOR", "identifier", "Many prey rows can belong to one predator record."],
        ["PREY", "`prey_name`, `tsn`, `qual_code`", "Prey label, ITIS TSN and life-stage/sex qualifier", "text/identifier", "Non-positive TSN values are sentinel or unresolved categories."],
        ["PREY", "`prey_length`, `ind_prey_wgt_g`", "Prey length and individual prey weight", "cm / g", "Both are stored as text and need parsing rules."],
        ["PREY", "`digestion`", "Source-specific digestion-state index", "index", "Documentation includes an unknown code; harmonise before modelling."],
        ["PREY", "`min_num`", "Recorded minimum prey quantity", "count-like", "A minimum/derived quantity, not always an exact integer count."],
        ["PREY", "`cpw`", "Calculated prey weight", "g", "Usually derived from mean mass or length-weight relationships, not directly observed biomass."],
        ["TAXONOMY", "`tsn`, `aphiaid`, hierarchy, functional groups", "ITIS/WoRMS identifiers and taxonomic/functional attributes", "identifier/categorical", "Taxonomic resolution and AphiaID coverage are incomplete."],
    ]

    table_count_rows = [
        [row["table_name"], fmt_int(row["row_count"])]
        for row in read_rows("table_row_counts")
    ]

    numeric_rows = [
        [
            row["variable"],
            row["unit"],
            fmt_int(row["non_null_rows"]),
            fmt_int(row["null_rows"]),
            f"{fmt_num(row['median'])} [{fmt_num(row['q1'])}, {fmt_num(row['q3'])}]",
            f"{fmt_num(row['mean'])} ± {fmt_num(row['sd'])}",
            fmt_num(row["p99"]),
            fmt_num(row["max"]),
            fmt_int(row["above_fence_rows"]),
        ]
        for row in read_rows("numeric_variable_summary")
    ]

    missing_candidates = []
    for row in read_rows("critical_field_missingness"):
        missing = as_int(row["null_rows"]) + as_int(row["blank_text_rows"])
        if missing:
            missing_candidates.append(
                (
                    missing / as_int(row["total_rows"]),
                    [
                        row["table_name"],
                        row["field_name"],
                        fmt_int(row["total_rows"]),
                        fmt_int(row["null_rows"]),
                        fmt_int(row["blank_text_rows"]),
                        pct(missing, row["total_rows"]),
                    ],
                )
            )
    missing_candidates.sort(key=lambda item: item[0], reverse=True)
    missing_rows = [row for _, row in missing_candidates[:20]]

    quality_explanations = {
        "duplicate_haul_id_rows": "HAUL primary-key duplicates",
        "duplicate_pred_id_rows": "PREDATOR primary-key duplicates",
        "duplicate_prey_id_rows": "PREY primary-key duplicates",
        "predator_orphan_haul_id": "PREDATOR rows without a matching HAUL",
        "prey_orphan_pred_id": "PREY rows without a matching PREDATOR",
        "haul_without_ices_row": "HAUL rows missing from the auxiliary ICES table",
        "num_empty_gt_num_stomachs": "`num_empty` greater than represented stomachs; inspect source",
        "negative_min_num": "Negative minimum prey quantities",
        "negative_cpw": "Negative calculated prey weights",
        "fractional_min_num": "Fractional `min_num`; retain as recorded pending interpretation",
        "fractional_num_empty": "Fractional `num_empty`; retain as recorded pending interpretation",
        "latitude_out_of_range": "Latitude outside [-90, 90]",
        "longitude_out_of_range": "Longitude outside [-180, 180]",
    }
    quality_rows = []
    for row in read_rows("data_quality_checks"):
        check = row["check_name"].strip()
        flagged = as_int(row["flagged_rows"])
        quality_rows.append(
            [
                check,
                fmt_int(flagged),
                "PASS" if flagged == 0 else "REVIEW",
                quality_explanations.get(check, "Review flagged records"),
            ]
        )

    limitation_rows = [
        ["Uneven effort", "Sampling varies strongly by sea, decade and source.", "Raw network size can track effort rather than ecology.", "Report hauls/stomachs with every network; use thresholds, offsets or rarefaction."],
        ["Pooled stomachs", f"{fmt_int(sum(as_int(r['predator_rows']) for r in pool if r['pooled']=='y'))} pooled rows represent {pct(sum(as_float(r['total_stomachs']) for r in pool if r['pooled']=='y'), total_stomachs)} of stomachs.", "A database row is not a uniform sampling unit.", "Weight/model by represented stomachs and run pooled-vs-individual sensitivity analyses."],
        ["Observation is not absence", "Unrecorded pairs can be unsampled or unresolved.", "Naive zeros and random negative sampling create trivial predictions.", "Construct pseudo-absences within comparable strata and evaluate grouped holdouts."],
        ["Taxonomic fragmentation", f"{fmt_int(raw_network['prey_names'])} prey labels map to {fmt_int(raw_network['prey_tsn'])} prey TSN.", "Name-based graphs inflate node and edge counts.", "Use TSN as node ID; preserve raw labels for traceability."],
        ["Non-positive prey TSN", f"{fmt_int(excluded_prey_rows)} rows ({pct(excluded_prey_rows, prey_rows)}) are excluded from the resolved-taxon primary graph.", "Empty/non-trophic and unresolved trophic evidence have different meanings.", "Separate empty/non-trophic codes; retain -99913 to -99915 in an unresolved-evidence sensitivity analysis."],
        ["Calculated weights", "`cpw` is calculated and method-dependent.", "It is not equivalent to directly observed biomass.", "Compare occurrence, `min_num` and `cpw` weightings."],
        ["Text-encoded numerics", "Depth and several length/weight fields are VARCHAR.", "Silent coercion can create missing or invalid values.", "Parse with explicit unit/domain rules and retain raw columns."],
        ["Version lineage", "The local file/schema is labelled 6.4 while the linked Cefas public description/report refers to 6.3.", "A publication could cite the wrong release.", "Identify the exact local snapshot by SHA-256 and confirm 6.4 release lineage with Cefas."],
    ]

    global_metric_rows = [
        ["Raw prey records", fmt_int(raw_network["prey_records"]), "All PREY rows linked to predators"],
        ["Resolved-taxon prey records", fmt_int(positive_network["prey_records"]), "`prey_tsn > 0`"],
        ["Excluded non-positive-TSN rows", fmt_int(excluded_prey_rows), "Kept separately for sampling/QC and sensitivity analyses"],
        ["Raw name-pair links", fmt_int(raw_name_edges), "Predator/prey labels; vulnerable to synonym fragmentation"],
        ["Resolved name-pair links", fmt_int(positive_name_edges), "Positive prey TSN but still grouped by labels"],
        ["Unified TSN nodes (S)", fmt_int(global_s), "Union of predator and prey TSN across primary networks"],
        ["Unified directed TSN links (L)", fmt_int(global_l), "Predator TSN → consumed prey TSN"],
        ["Self-links", fmt_int(global_self_loops), "Potential cannibalism; retained and reported separately"],
        ["Primary non-self density", f"{global_density:.4f}", "(L - self-links) / [S(S - 1)]"],
        ["Sensitivity connectance", f"{global_connectance:.4f}", "L / S², including self-link opportunities"],
    ]

    stratum_rows = [
        [
            row["sea"],
            row["decade"],
            fmt_int(row["hauls"]),
            fmt_int(row["taxon_nodes"]),
            fmt_int(row["directed_edges"]),
            f"{as_float(row['nonself_directed_density']):.4f}",
            fmt_num(row["links_per_taxon"]),
            fmt_int(row["weak_components"]),
            f"{as_float(row['largest_component_pct']):.1f}%",
        ]
        for row in network_metrics[:12]
    ]

    representative_rows = [
        [
            row["complexity_tier"].capitalize(),
            row["sea"],
            row["decade"],
            fmt_int(row["hauls"]),
            fmt_int(row["taxon_nodes"]),
            fmt_int(row["directed_edges"]),
            f"{as_float(row['nonself_directed_density']):.4f}",
            fmt_num(row["links_per_taxon"]),
            f"{as_float(row['top_10_edge_record_share_pct']):.1f}%",
        ]
        for row in representatives
    ]
    densest_rep = max(representatives, key=lambda row: row["nonself_directed_density"])
    most_concentrated_rep = max(
        representatives, key=lambda row: row["top_10_edge_record_share_pct"]
    )

    validation_rows = [
        [
            row["check_name"],
            row["status"],
            row["observed"],
            row["expected"],
            row["detail"],
        ]
        for row in read_rows("validation_checks")
    ]

    research_question_rows = [
        ["RQ1", "How do resolved food-web size, density, degree structure and components vary among seas and decades after controlling for effort?"],
        ["RQ2", "How robust are inferred structures to spatial grain, temporal binning, pooled records and edge weighting (`occurrence`, `min_num`, `cpw`)?"],
        ["RQ3", "Which predator traits, prey taxonomy and environmental/context variables explain observed interactions and diet breadth?"],
        ["RQ4", "Can link-prediction models recover held-out interactions in new campaigns, decades, seas or predators better than simple baselines?"],
        ["RQ5", "How much do provenance, taxonomic resolution and non-positive/unresolved prey codes change ecological conclusions?"],
    ]

    preprocessing_rows = [
        ["1. Freeze and trace", "Record source path, checksum, extraction date, schema and software versions; never overwrite the Access source."],
        ["2. Validate keys", "Enforce HAUL → PREDATOR → PREY and PROVENANCE links; review auxiliary ICES gaps and duplicated IDs."],
        ["3. Parse fields", "Convert text-encoded depth/length/weight with explicit missing tokens, units and domain checks; retain raw values."],
        ["4. Normalise taxonomy", "Use TSN as primary node ID, attach canonical name/AphiaID, preserve qualifiers, and log synonym merges."],
        ["5. Classify evidence", "Separate empty/non-trophic codes, resolved positive TSN, and unresolved trophic codes (-99913 to -99915)."],
        ["6. Define sampling unit", "Keep haul, campaign, pooling and represented-stomach fields; do not treat every predator row as equal effort."],
        ["7. Construct edges", "Aggregate within declared sea × decade (then test alternative grains); retain occurrence, `min_num`, `cpw` and distinct-haul support."],
        ["8. Apply eligibility", "Set minimum effort/taxon support before comparisons; record excluded strata and run threshold sensitivity analyses."],
        ["9. Build evaluation splits", "Hold out campaigns/time/space/predators as groups and sample negatives only within ecologically comparable candidate sets."],
    ]

    analysis_rows = [
        ["Network description", "S, L, self-links, non-self density, L/S, in/out/total degree, weighted strength, components, largest-component share, basal/intermediate/top fractions."],
        ["Structural comparison", "Effort-aware comparisons across sea/decade; rarefaction or coverage standardisation; sensitivity to graph grain and edge weights."],
        ["Community/trophic structure", "Modularity, motifs, trophic position/height and generality/vulnerability where taxonomic resolution supports them."],
        ["Composition", "Ordination and permutation tests for diet/link composition, with campaign/sea/decade restrictions and effort-aware distances."],
        ["Statistical models", "GLMM/GAM or hierarchical models for edge presence/weight and network metrics; random effects for campaign/predator and nonlinear time trends."],
        ["Link prediction", "Degree/common-neighbour heuristics, trait baselines, WLNM/SEAL variants and calibrated classifiers evaluated on grouped holdouts."],
        ["Sensitivity/null models", "Pooling, unresolved prey, taxonomic aggregation, effort thresholds, weighting schemes and degree-preserving/ecological null models."],
    ]

    risk_rows = [
        ["Observed links represent realised diet; non-links are not confirmed absences.", "False negatives and inflated model performance.", "Use constrained pseudo-absences and grouped external-style holdouts."],
        ["Sampling effort and source protocols are exchangeable after adjustment.", "Residual confounding across time/space/source.", "Include effort/provenance covariates and stratified sensitivity analyses."],
        ["TSN resolves biological identity consistently through time.", "Synonyms, aggregated prey and population labels distort nodes.", "Audit mappings; publish raw-to-canonical crosswalk and resolution flags."],
        ["`min_num` and `cpw` are comparable across source formats.", "Weight-based trends may reflect derivation methods.", "Analyse occurrence, counts and calculated weight separately."],
        ["Sea × decade is an ecologically meaningful first grain.", "Aggregation can hide seasonality and local structure.", "Repeat at division/rectangle/season where sample size permits."],
        ["Random ML splits are representative.", "Leakage through campaign, predator or repeated edges.", "Prohibit random-edge-only claims; use grouped and temporal/spatial transfer tests."],
    ]

    output_rows = [
        ["Versioned extraction manifest", "Source checksum, schema, query log and generated-file inventory.", "Checksum match; zero extractor errors; deterministic rerun."],
        ["Clean relational tables and crosswalks", "Typed fields, retained raw values, key/taxonomy/provenance mappings.", "Unique PKs, resolved FKs, documented parsing failures and reconciliation totals."],
        ["Edge tables by declared stratum", "TSN-to-TSN directed links with occurrence, haul support, `min_num`, `cpw`, effort and provenance.", "No non-positive prey TSN in primary graph; raw = retained + excluded; duplicate aggregation checked."],
        ["Network/node metrics", "Graph-level and node-level metrics with definitions and loop policy.", "Metric bounds; sum in-degree = sum out-degree = L; sensitivity definitions agree."],
        ["EDA figures", "Coverage, distributions, missingness, effort and representative individual networks.", "Selection/layout deterministic; legends, units and filters visible."],
        ["Statistical/model results", "Effect estimates, uncertainty, predictions, grouped splits and baselines.", "No group leakage; calibration/discrimination reported; outperform declared baselines."],
        ["Sensitivity and QC report", "Pooling, taxonomy, weighting, grain, threshold and unresolved-evidence analyses.", "Conclusions labelled robust or conditional; all review flags resolved or justified."],
    ]

    implementation_rows = [
        ["Phase 1 — Provenance and dictionary", "Freeze snapshot, citation/version note, schema and units.", "Manifest, checksum and compact data dictionary complete."],
        ["Phase 2 — QC and preprocessing", "Keys, missingness, parseability, semantic anomalies, taxonomy and evidence classes.", "QC report passes or every exception has a recorded decision."],
        ["Phase 3 — Graph construction", "Create reproducible primary and sensitivity edge lists by stratum.", "Counts reconcile and graph validation identities pass."],
        ["Phase 4 — EDA and network metrics", "Describe effort, distributions, topology and representative graphs.", "All figures/tables regenerate and comparisons include effort."],
        ["Phase 5 — Statistical and link-prediction models", "Fit baselines and candidate models with grouped splits.", "Pre-registered metrics, uncertainty and leakage checks complete."],
        ["Phase 6 — Sensitivity and synthesis", "Repeat across pooling, weights, grain, taxonomy and thresholds.", "Final claims distinguish robust findings from assumption-dependent findings."],
    ]

    cells = [
        markdown_cell(
            "# DAPSTOM 6.4 Local Snapshot — Dataset Explanation, EDA and Analysis Plan\n\n"
            "This notebook documents the exact Microsoft Access snapshot supplied as CEFAS/DAPSTOM data, "
            "summarises its analytical content and limitations, and defines a reproducible analysis plan. "
            "The companion `DAPSTOM_EDA_visuals.ipynb` contains distributions and representative individual networks."
        ),
        markdown_cell(
            "## Executive Summary\n\n"
            f"- Relational chain: `PROVENANCE → HAUL → PREDATOR → PREY`, with taxonomy linked by TSN.\n"
            f"- Scale: **{hauls:,} hauls**, **{predator_rows:,} predator records**, "
            f"**{total_stomachs:,.0f} represented stomachs**, and **{prey_rows:,} prey records**.\n"
            f"- Coverage: **{min_year}–{max_year}**, with uneven spatial, temporal and source effort.\n"
            f"- Primary resolved network rule: `prey_tsn > 0`, TSN node identity, directed relation "
            "`predator → consumed prey`, self-links retained but reported separately.\n"
            f"- Primary graph across all strata: **S={global_s:,} TSN nodes**, **L={global_l:,} links**, "
            f"non-self density **{global_density:.4f}**.\n"
            "- The dataset supports food-web and link-prediction work, but comparisons must account for "
            "pooling, effort, provenance, taxonomic resolution and calculated rather than observed weights."
        ),
        markdown_cell(
            "## 1. Scope, Provenance and Citation\n\n"
            f"The analysed object is the local file labelled `DAPSTOM 6-4 COMBINED`, size "
            f"**{manifest['source_size_bytes']:,} bytes**, SHA-256 "
            f"`{manifest['source_sha256']}`. The original supplied file and working copy are byte-identical "
            "(see validation table below), and extraction is read-only.\n\n"
            "Custodian and official record: **Centre for Environment, Fisheries & Aquaculture Science "
            "(Cefas)**, [DOI 10.14466/CefasDataHub.144]"
            "(https://doi.org/10.14466/CefasDataHub.144), listed under the Open Government Licence. "
            "The public description attributes the initiative to Defra/EU support and describes digitised "
            "fish-stomach records from logbooks/reports, partner contributions and publications.\n\n"
            "**Version note:** the local file and schema are labelled **6.4**, while the available public "
            "description/report refers to **6.3**. The report also gives 481,476 stomachs, whereas the local "
            f"snapshot sums to {total_stomachs:,.0f}. Therefore this analysis cites the exact checksum and treats "
            "the 6.4-to-6.3 release lineage as a Cefas confirmation item before publication.\n\n"
            f"The `PROVENANCE` table contains **{provenance_rows:,} campaigns/sources** and is complete for its "
            "five inspected fields:\n\n"
            + md_table(["Source type", "Campaign/source rows", "Linked hauls"], provenance_table)
            + "\n\nThe auxiliary `cefas_arctic_cruises_and_catches_1930_1959.csv` is **outside the scope** "
            "of this Access-database EDA and is not silently merged."
        ),
        markdown_cell(
            "## 2. Relational Model, Variables, Units and Identifiers\n\n"
            "Row meaning:\n\n"
            "- `HAUL`: a sampling event/location.\n"
            "- `PREDATOR`: an individual or pooled stomach record; one row is not necessarily one stomach.\n"
            "- `PREY`: one prey item/category attached to a predator record; multiple rows may share `pred_id`.\n"
            "- `PROVENANCE`: campaign/source documentation linked by `cruise_name`.\n\n"
            "`PROVENANCE.cruise_name → HAUL.cruise_name →(haul_id) PREDATOR "
            "→(pred_id) PREY`; TSN links predator/prey records to taxonomy.\n\n"
            + md_table(["Table", "Variables", "Meaning / key role", "Unit or type", "Analytical note"], variable_rows)
        ),
        markdown_cell(
            "## 3. Ecological and Analytical Meaning\n\n"
            "An observed DAPSTOM edge is evidence that a prey category occurred in the realised diet of a sampled "
            "predator under a particular place, time, body size, campaign and recording protocol. The selected "
            "database direction is **predator → consumed prey**; ecological energy flow is the reverse. "
            "An unobserved edge is not evidence of ecological absence.\n\n"
            "`min_num` is a recorded minimum quantity and may encode source-derived evidence; `cpw` approximates "
            "prey mass using calculation rules and should not be presented as uniformly observed biomass. "
            "Ontogeny, pooling, sampling effort and provenance can all alter apparent diet and network structure."
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
            "manifest = json.loads((TABLE_DIR / 'eda_manifest.json').read_text())\n"
            "sorted(path.name for path in TABLE_DIR.glob('*.csv'))"
        ),
        markdown_cell(
            "## 4. Dataset Dimensions and Coverage\n\n"
            + md_table(["Table", "Rows"], table_count_rows)
            + "\n\nSpatial coverage by available resolution:\n\n"
            + md_table(["Resolution", "Hauls", "Coverage"], spatial_rows)
            + "\n\nLargest sea-level groups:\n\n"
            + md_table(
                ["Sea", "Hauls", "With lat/lon", "With ICES rectangle", "Distinct rectangles"],
                sea_rows,
            )
        ),
        markdown_cell(
            "## 5. Sampling Unit: Individual and Pooled Stomachs\n\n"
            + md_table(
                [
                    "Pooled",
                    "Predator rows",
                    "% rows",
                    "Represented stomachs",
                    "% stomachs",
                    "Mean stomachs/row",
                ],
                pool_rows,
            )
            + "\n\nPooled rows are a minority of database rows but dominate represented stomachs. "
            "Network comparisons must therefore report both row counts and effort expressed as hauls/stomachs."
        ),
        markdown_cell(
            "## 6. Descriptive Statistics, Distributions and Outliers\n\n"
            "Tukey fences (`Q1 - 1.5×IQR`, `Q3 + 1.5×IQR`) are used as **review flags**, not automatic deletion rules. "
            "Heavy tails are expected for pooled effort and dietary quantities. The companion visual notebook shows "
            "the corresponding continuous-variable histograms (log-transformed where appropriate).\n\n"
            + md_table(
                [
                    "Variable",
                    "Unit",
                    "Non-null n",
                    "Missing",
                    "Median [Q1, Q3]",
                    "Mean ± SD",
                    "P99",
                    "Max",
                    "Above Tukey fence",
                ],
                numeric_rows,
            )
        ),
        markdown_cell(
            "## 7. Missingness and Data-Quality Checks\n\n"
            "Highest missingness rates among inspected analytical fields:\n\n"
            + md_table(
                ["Table", "Field", "Rows", "Null", "Blank text", "Missing rate"],
                missing_rows,
            )
            + "\n\nIntegrity and domain checks:\n\n"
            + md_table(["Check", "Flagged rows", "Status", "Interpretation/action"], quality_rows)
            + "\n\nRows marked `REVIEW` remain in the source-derived summaries until their semantics are checked "
            "against provenance; they are not silently corrected or deleted."
        ),
        markdown_cell(
            "## 8. Known Limitations and Required Controls\n\n"
            + md_table(["Issue", "Evidence", "Risk", "Control / mitigation"], limitation_rows)
        ),
        markdown_cell(
            "## 9. Network Definition and Global Properties\n\n"
            "Primary resolved network:\n\n"
            "- Node identity: unified positive TSN across predator and prey roles.\n"
            "- Edge: `predator_tsn → prey_tsn`, aggregated within `sea × decade`.\n"
            "- Self-links: retained as potential cannibalism, counted separately, removed from the denominator of "
            "the primary density `L_nonself/[S(S−1)]`.\n"
            "- Sensitivity connectance: `L/S²`, allowing self-link opportunities.\n"
            "- Edge attributes: prey-record frequency, distinct-haul support, summed `min_num` and summed `cpw`.\n\n"
            + md_table(["Property", "Value", "Definition / caveat"], global_metric_rows)
        ),
        markdown_cell(
            "## 10. Sea × Decade Network Metrics\n\n"
            f"Metrics were generated for **{len(network_metrics)}** resolved-taxon sea-decade networks. "
            "These values describe complete stratum graphs, not only the readable backbones shown in the visual notebook.\n\n"
            + md_table(
                [
                    "Sea",
                    "Decade",
                    "Hauls",
                    "S",
                    "L",
                    "Non-self density",
                    "L/S",
                    "Weak components",
                    "Largest component",
                ],
                stratum_rows,
            )
        ),
        markdown_cell(
            "## 11. Representative Individual Networks and Structural Differences\n\n"
            "Selection is deterministic: among strata with at least 30 hauls, 10 predator taxa and 100 prey taxa, "
            "choose high, median and low directed-edge complexity while preferring different seas. Metrics refer to "
            "each **complete** graph; visual figures show a labelled high-support backbone for readability.\n\n"
            + md_table(
                [
                    "Complexity tier",
                    "Sea",
                    "Decade",
                    "Hauls",
                    "S",
                    "L",
                    "Density",
                    "L/S",
                    "Top-10 record share",
                ],
                representative_rows,
            )
            + f"\n\nAmong these examples, **{densest_rep['sea']} {densest_rep['decade']}** has the highest "
            f"non-self density ({densest_rep['nonself_directed_density']:.4f}), while "
            f"**{most_concentrated_rep['sea']} {most_concentrated_rep['decade']}** has the greatest concentration "
            f"of prey records in its ten strongest links ({most_concentrated_rep['top_10_edge_record_share_pct']:.1f}%). "
            "These contrasts are descriptive: graph structure and sampling effort/provenance remain entangled."
        ),
        markdown_cell(
            "## 12. Main Research Questions\n\n"
            + md_table(["ID", "Research question"], research_question_rows)
        ),
        markdown_cell(
            "## 13. Required Preprocessing\n\n"
            + md_table(["Order", "Operational rule"], preprocessing_rows)
        ),
        markdown_cell(
            "## 14. Candidate Network and Statistical Analyses\n\n"
            + md_table(["Analysis family", "Candidate analyses"], analysis_rows)
        ),
        markdown_cell(
            "## 15. Assumptions, Risks and Mitigations\n\n"
            + md_table(["Assumption / risk", "Potential impact", "Mitigation"], risk_rows)
        ),
        markdown_cell(
            "## 16. Expected Outputs and Acceptance Checks\n\n"
            + md_table(["Output", "Required contents", "Validation / acceptance"], output_rows)
        ),
        markdown_cell(
            "## 17. Current Automated Validation\n\n"
            + md_table(["Check", "Status", "Observed", "Expected", "Meaning"], validation_rows)
            + "\n\n`REVIEW` is not an extraction failure: it identifies source-level semantic anomalies requiring "
            "a documented decision before modelling."
        ),
        markdown_cell(
            "## 18. Realistic Implementation Sequence\n\n"
            + md_table(["Phase", "Work", "Completion gate"], implementation_rows)
        ),
        markdown_cell(
            "## 19. Reproducibility and Generated Files\n\n"
            "From the repository root:\n\n"
            "```bash\n"
            "bash data/processed/dapstom_eda/tools/run_extractor.sh\n"
            "python3 data/processed/dapstom_eda/tools/derive_network_metrics.py\n"
            "python3 data/processed/dapstom_eda/tools/build_notebook.py\n"
            "python3 data/processed/dapstom_eda/tools/build_visual_notebook.py\n"
            "```\n\n"
            "Key generated artifacts:\n\n"
            "- `tables/eda_manifest.json`: timestamp, source path, size, checksum and extraction driver.\n"
            "- `tables/numeric_variable_summary.csv` and `numeric_variable_histograms.csv`: continuous-variable EDA.\n"
            "- `tables/data_quality_checks.csv` and `validation_checks.csv`: QC and invariant checks.\n"
            "- `tables/sea_decade_network_edges_positive_prey_tsn.csv`: filtered stratum edge evidence.\n"
            "- `tables/sea_decade_network_metrics.csv` and `sea_decade_node_degrees.csv`: graph/node metrics.\n"
            "- `tables/representative_networks.csv`: deterministic graph selection.\n"
            "- `figures/*.svg`: reproducible visual EDA and representative networks.\n\n"
            "The saved Access view `Query1` emits a known non-blocking duplicate-column warning; extraction uses "
            "base tables with explicit aliases, and `query_errors.csv` must remain empty."
        ),
    ]

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
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    build()
