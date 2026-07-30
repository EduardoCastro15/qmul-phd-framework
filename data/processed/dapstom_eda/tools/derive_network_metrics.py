import csv
import filecmp
import json
from collections import defaultdict
from pathlib import Path


EDA_DIR = Path(__file__).resolve().parents[1]
TABLE_DIR = EDA_DIR / "tables"
REPO_ROOT = EDA_DIR.parents[2]
ORIGINAL_ACCDB = (
    REPO_ROOT
    / "data"
    / "DAPSTOM – An Integrated Database & Portal for Fish Stomach Records"
    / "DAPSTOM 6-4 COMBINED.accdb"
)
WORKING_ACCDB = REPO_ROOT / "data" / "dapstom_6_4_combined_working_copy.accdb"

MIN_REPRESENTATIVE_HAULS = 30
MIN_REPRESENTATIVE_PREDATOR_TAXA = 10
MIN_REPRESENTATIVE_PREY_TAXA = 100
REPRESENTATIVE_COUNT = 3


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


def normalise_tsn(value):
    if value in (None, ""):
        return ""
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(numeric)


def stratum_key(row):
    sea = row.get("sea") or row.get("SEA") or "(missing sea)"
    decade = as_int(row.get("decade") or row.get("DECADE"))
    return sea, decade


def collapse_network_edges():
    grouped = defaultdict(dict)
    for row in read_rows("sea_decade_network_edges_positive_prey_tsn"):
        key = stratum_key(row)
        predator_tsn = normalise_tsn(row["predator_tsn"])
        prey_tsn = normalise_tsn(row["prey_tsn"])
        if not predator_tsn or not prey_tsn:
            continue
        edge_key = predator_tsn, prey_tsn
        current = grouped[key].setdefault(
            edge_key,
            {
                "predator_tsn": predator_tsn,
                "prey_tsn": prey_tsn,
                "predator_name": row["predator_name"],
                "prey_name": row["prey_name"],
                "prey_records": 0,
                "distinct_hauls": 0,
                "total_min_num": 0.0,
                "total_cpw": 0.0,
            },
        )
        current["prey_records"] += as_int(row["prey_records"])
        current["distinct_hauls"] += as_int(row.get("distinct_hauls"))
        current["total_min_num"] += as_float(row["total_min_num"])
        current["total_cpw"] += as_float(row["total_cpw"])
        if len(row["predator_name"]) > len(current["predator_name"]):
            current["predator_name"] = row["predator_name"]
        if len(row["prey_name"]) > len(current["prey_name"]):
            current["prey_name"] = row["prey_name"]
    return {
        key: sorted(edges.values(), key=lambda edge: edge["prey_records"], reverse=True)
        for key, edges in grouped.items()
    }


def effort_by_stratum():
    effort = {}
    for row in read_rows("network_potential_by_sea_decade_positive_prey_tsn"):
        effort[stratum_key(row)] = {
            "hauls": as_int(row["hauls"]),
            "predator_records": as_int(row["predator_records"]),
            "prey_records": as_int(row["prey_records"]),
        }
    return effort


def component_summary(nodes, edges):
    parent = {node: node for node in nodes}

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != node:
            next_node = parent[node]
            parent[node] = root
            node = next_node
        return root

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in edges:
        union(edge["predator_tsn"], edge["prey_tsn"])

    sizes = defaultdict(int)
    for node in nodes:
        sizes[find(node)] += 1
    largest = max(sizes.values(), default=0)
    return len(sizes), largest


def network_metrics(key, edges, effort):
    predators = {edge["predator_tsn"] for edge in edges}
    prey = {edge["prey_tsn"] for edge in edges}
    nodes = predators | prey
    edge_count = len(edges)
    self_loops = sum(edge["predator_tsn"] == edge["prey_tsn"] for edge in edges)
    nonself_edges = edge_count - self_loops
    total_records = sum(edge["prey_records"] for edge in edges)
    top_ten_records = sum(edge["prey_records"] for edge in edges[:10])
    components, largest_component = component_summary(nodes, edges)
    node_count = len(nodes)
    out_degree = defaultdict(int)
    in_degree = defaultdict(int)
    for edge in edges:
        out_degree[edge["predator_tsn"]] += 1
        in_degree[edge["prey_tsn"]] += 1
    connectance = edge_count / (node_count * node_count) if node_count else 0.0
    nonself_density = (
        nonself_edges / (node_count * (node_count - 1))
        if node_count > 1
        else 0.0
    )
    sea, decade = key
    sampling = effort.get(key, {})
    return {
        "sea": sea,
        "decade": decade,
        "hauls": sampling.get("hauls", 0),
        "predator_records": sampling.get("predator_records", 0),
        "prey_records": sampling.get("prey_records", total_records),
        "taxon_nodes": node_count,
        "predator_taxa": len(predators),
        "prey_taxa": len(prey),
        "directed_edges": edge_count,
        "self_loops": self_loops,
        "connectance_l_over_s2": connectance,
        "nonself_directed_density": nonself_density,
        "links_per_taxon": edge_count / node_count if node_count else 0.0,
        "mean_prey_taxa_per_predator": edge_count / len(predators) if predators else 0.0,
        "mean_predators_per_prey_taxon": edge_count / len(prey) if prey else 0.0,
        "max_out_degree": max(out_degree.values(), default=0),
        "max_in_degree": max(in_degree.values(), default=0),
        "mean_total_degree": 2 * edge_count / node_count if node_count else 0.0,
        "max_total_degree": max(
            (out_degree[node] + in_degree[node] for node in nodes),
            default=0,
        ),
        "weak_components": components,
        "largest_component_taxa": largest_component,
        "largest_component_pct": 100 * largest_component / node_count if node_count else 0.0,
        "top_10_edge_record_share_pct": (
            100 * top_ten_records / total_records if total_records else 0.0
        ),
    }


def select_representatives(metrics):
    eligible = [
        row
        for row in metrics
        if row["hauls"] >= MIN_REPRESENTATIVE_HAULS
        and row["predator_taxa"] >= MIN_REPRESENTATIVE_PREDATOR_TAXA
        and row["prey_taxa"] >= MIN_REPRESENTATIVE_PREY_TAXA
    ]
    eligible.sort(
        key=lambda row: (
            row["directed_edges"],
            row["hauls"],
            row["prey_records"],
        ),
        reverse=True,
    )
    selected = []
    used_seas = set()
    target_indices = [0, len(eligible) // 2, max(0, len(eligible) - 1)]
    tier_names = ["high", "median", "low"]
    for target, tier in zip(target_indices, tier_names):
        candidates = [
            (abs(index - target), index, row)
            for index, row in enumerate(eligible)
            if row not in selected and row["sea"] not in used_seas
        ]
        if not candidates:
            candidates = [
                (abs(index - target), index, row)
                for index, row in enumerate(eligible)
                if row not in selected
            ]
        if not candidates:
            break
        _, _, chosen = min(candidates, key=lambda item: (item[0], item[1]))
        selected.append({**chosen, "complexity_tier": tier})
        used_seas.add(chosen["sea"])
    return selected


def degree_rows(key, edges):
    sea, decade = key
    node_names = {}
    out_neighbors = defaultdict(set)
    in_neighbors = defaultdict(set)
    out_strength = defaultdict(int)
    in_strength = defaultdict(int)
    nodes = set()
    for edge in edges:
        predator = edge["predator_tsn"]
        prey = edge["prey_tsn"]
        nodes.update([predator, prey])
        node_names.setdefault(predator, edge["predator_name"])
        if len(edge["prey_name"]) > len(node_names.get(prey, "")):
            node_names[prey] = edge["prey_name"]
        out_neighbors[predator].add(prey)
        in_neighbors[prey].add(predator)
        out_strength[predator] += edge["prey_records"]
        in_strength[prey] += edge["prey_records"]
    return [
        {
            "sea": sea,
            "decade": decade,
            "tsn": node,
            "display_name": node_names.get(node, node),
            "out_degree": len(out_neighbors[node]),
            "in_degree": len(in_neighbors[node]),
            "total_degree": len(out_neighbors[node]) + len(in_neighbors[node]),
            "out_record_strength": out_strength[node],
            "in_record_strength": in_strength[node],
        }
        for node in sorted(nodes, key=lambda value: as_float(value))
    ]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_validation_checks(networks, metrics, representatives, degree_table):
    checks = []

    def add(check_name, status, observed, expected, detail):
        checks.append(
            {
                "check_name": check_name,
                "status": status,
                "observed": observed,
                "expected": expected,
                "detail": detail,
            }
        )

    invalid_primary_edges = sum(
        as_float(edge["prey_tsn"]) <= 0
        for edges in networks.values()
        for edge in edges
    )
    add(
        "primary_network_positive_prey_tsn",
        "PASS" if invalid_primary_edges == 0 else "FAIL",
        invalid_primary_edges,
        0,
        "Primary sea-decade networks must exclude non-positive prey TSN codes.",
    )

    invalid_metric_rows = sum(
        row["directed_edges"] > row["taxon_nodes"] * row["taxon_nodes"]
        or not 0 <= row["connectance_l_over_s2"] <= 1
        or not 0 <= row["nonself_directed_density"] <= 1
        for row in metrics
    )
    add(
        "network_metric_bounds",
        "PASS" if invalid_metric_rows == 0 else "FAIL",
        invalid_metric_rows,
        0,
        "Every graph must satisfy L <= S^2 and both reported density measures must lie in [0, 1].",
    )

    degree_by_stratum = defaultdict(lambda: {"in": 0, "out": 0})
    for row in degree_table:
        key = row["sea"], as_int(row["decade"])
        degree_by_stratum[key]["in"] += as_int(row["in_degree"])
        degree_by_stratum[key]["out"] += as_int(row["out_degree"])
    degree_failures = 0
    for row in metrics:
        key = row["sea"], as_int(row["decade"])
        degree_failures += (
            degree_by_stratum[key]["in"] != row["directed_edges"]
            or degree_by_stratum[key]["out"] != row["directed_edges"]
        )
    add(
        "degree_edge_conservation",
        "PASS" if degree_failures == 0 else "FAIL",
        degree_failures,
        0,
        "For each directed network, sum(in-degree) = sum(out-degree) = L.",
    )

    raw = read_rows("global_network_summary")[0]
    positive = read_rows("global_network_summary_positive_prey_tsn")[0]
    raw_records = as_int(raw["prey_records"])
    positive_records = as_int(positive["prey_records"])
    excluded_records = raw_records - positive_records
    add(
        "prey_record_filter_accounting",
        "PASS" if raw_records == positive_records + excluded_records else "FAIL",
        f"{raw_records} = {positive_records} + {excluded_records}",
        "raw = retained + excluded",
        "The non-positive-TSN filter must reconcile exactly to the raw prey-record total.",
    )

    add(
        "representative_network_count",
        "PASS" if len(representatives) == REPRESENTATIVE_COUNT else "FAIL",
        len(representatives),
        REPRESENTATIVE_COUNT,
        "The deterministic high/median/low selection should produce three eligible networks.",
    )

    query_errors = read_rows("query_errors")
    add(
        "extractor_query_errors",
        "PASS" if not query_errors else "FAIL",
        len(query_errors),
        0,
        "All Access extraction queries must complete without recorded errors.",
    )

    quality = {row["check_name"].strip(): as_int(row["flagged_rows"]) for row in read_rows("data_quality_checks")}
    integrity_checks = [
        "duplicate_haul_id_rows",
        "duplicate_pred_id_rows",
        "duplicate_prey_id_rows",
        "predator_orphan_haul_id",
        "prey_orphan_pred_id",
    ]
    integrity_flags = sum(quality.get(name, 0) for name in integrity_checks)
    add(
        "key_uniqueness_and_referential_integrity",
        "PASS" if integrity_flags == 0 else "FAIL",
        integrity_flags,
        0,
        "Primary identifiers must be unique and HAUL -> PREDATOR -> PREY foreign keys must resolve.",
    )

    ices_gaps = quality.get("haul_without_ices_row", 0)
    add(
        "auxiliary_ices_table_coverage",
        "REVIEW" if ices_gaps else "PASS",
        ices_gaps,
        0,
        "The auxiliary ICES table should be reconciled before ICES-specific analyses.",
    )

    semantic_flags = sum(
        quality.get(name, 0)
        for name in [
            "num_empty_gt_num_stomachs",
            "fractional_min_num",
            "fractional_num_empty",
        ]
    )
    add(
        "semantic_count_anomalies",
        "REVIEW" if semantic_flags else "PASS",
        semantic_flags,
        0,
        "Flagged rows require source-level review; they are not deleted automatically.",
    )

    source_files_match = (
        ORIGINAL_ACCDB.exists()
        and WORKING_ACCDB.exists()
        and filecmp.cmp(ORIGINAL_ACCDB, WORKING_ACCDB, shallow=False)
    )
    manifest = json.loads((TABLE_DIR / "eda_manifest.json").read_text(encoding="utf-8"))
    add(
        "source_and_working_copy_match",
        "PASS" if source_files_match else "FAIL",
        manifest.get("source_sha256", ""),
        "identical byte content",
        "The original supplied Access file and the read-only working copy must be byte-identical.",
    )

    write_csv(
        TABLE_DIR / "validation_checks.csv",
        ["check_name", "status", "observed", "expected", "detail"],
        checks,
    )


def derive():
    networks = collapse_network_edges()
    effort = effort_by_stratum()
    metrics = [
        network_metrics(key, edges, effort)
        for key, edges in networks.items()
    ]
    metrics.sort(
        key=lambda row: (
            row["directed_edges"],
            row["hauls"],
            row["sea"],
            row["decade"],
        ),
        reverse=True,
    )
    metric_fields = [
        "sea",
        "decade",
        "hauls",
        "predator_records",
        "prey_records",
        "taxon_nodes",
        "predator_taxa",
        "prey_taxa",
        "directed_edges",
        "self_loops",
        "connectance_l_over_s2",
        "nonself_directed_density",
        "links_per_taxon",
        "mean_prey_taxa_per_predator",
        "mean_predators_per_prey_taxon",
        "max_out_degree",
        "max_in_degree",
        "mean_total_degree",
        "max_total_degree",
        "weak_components",
        "largest_component_taxa",
        "largest_component_pct",
        "top_10_edge_record_share_pct",
    ]
    write_csv(TABLE_DIR / "sea_decade_network_metrics.csv", metric_fields, metrics)
    all_degree_rows = []
    for key, edges in networks.items():
        all_degree_rows.extend(degree_rows(key, edges))
    write_csv(
        TABLE_DIR / "sea_decade_node_degrees.csv",
        [
            "sea",
            "decade",
            "tsn",
            "display_name",
            "out_degree",
            "in_degree",
            "total_degree",
            "out_record_strength",
            "in_record_strength",
        ],
        all_degree_rows,
    )

    representatives = select_representatives(metrics)
    representative_rows = []
    for rank, row in enumerate(representatives, start=1):
        representative_rows.append(
            {
                "selection_rank": rank,
                **row,
                "selection_rule": (
                    f"{row['complexity_tier'].capitalize()} directed-edge complexity among "
                    "eligible strata, preferring a distinct sea; eligibility requires "
                    f">= {MIN_REPRESENTATIVE_HAULS} hauls, "
                    f">= {MIN_REPRESENTATIVE_PREDATOR_TAXA} predator taxa and "
                    f">= {MIN_REPRESENTATIVE_PREY_TAXA} prey taxa."
                ),
            }
        )
    write_csv(
        TABLE_DIR / "representative_networks.csv",
        ["selection_rank", *metric_fields, "complexity_tier", "selection_rule"],
        representative_rows,
    )
    write_validation_checks(networks, metrics, representative_rows, all_degree_rows)
    return networks, metrics, representative_rows


if __name__ == "__main__":
    _, metrics, representatives = derive()
    print(
        f"Wrote metrics for {len(metrics)} sea-decade networks and selected "
        f"{len(representatives)} representatives."
    )
