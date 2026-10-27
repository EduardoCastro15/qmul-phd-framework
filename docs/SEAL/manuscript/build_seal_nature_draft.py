#!/usr/bin/env python3
"""Build the provisional SEAL Nature Ecology & Evolution manuscript.

The original scaffold is the formatting authority. New text is inserted by
cloning its existing title, body, Heading 1 and Heading 2 paragraphs; no new
styles or highlighting are introduced. ``Acknowledgements`` and every
subsequent paragraph are copied byte-for-byte from the original document XML.
All other package parts, including Mendeley webextensions, styles, footnotes,
endnotes and relationships, are also copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


DEFAULT_DOCX = Path(
    "/Users/acw792/Library/CloudStorage/OneDrive-QueenMary,UniversityofLondon/"
    "QMUL/PhD/Stage 3/Reports/Jorge Stage 3 manuscript - Oct 27 JC.docx"
)
BACKUP_SUFFIX = ".before_seal_draft_2026-10-27.docx"
CORRECTION_BACKUP_SUFFIX = ".before_style_scope_correction_2026-10-27.docx"
DOCUMENT_XML = "word/document.xml"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

MARKER_RE = re.compile(
    r"(\{CITE:[^}]+\}|"
    r"\[(?:IMPLEMENTED—VERIFY PROVENANCE|PLANNED—NOT YET IMPLEMENTED|"
    r"RESULT REQUIRED|AUTHOR CONFIRMATION REQUIRED|VERIFY DATASET CITATION|"
    r"DOI/REPOSITORY PENDING|CORRESPONDING AUTHOR EMAIL REQUIRED|"
    r"REPORTING SUMMARY REQUIRED)[^\]]*\])"
)


ABSTRACT = (
    "Species interaction networks are necessarily incomplete, yet distinguishing unobserved "
    "from ecologically implausible links is essential for forecasting community responses. "
    "We develop a direction-aware extension of subgraph-based link prediction for trophic "
    "networks that represents resource and consumer roles separately and integrates node "
    "traits, missingness indicators and contextual edge covariates. The study is designed to "
    "compare this model with original and directed SEAL baselines using matched splits across "
    "290 food webs, network-disjoint and leave-one-ecosystem-out evaluation, and rolling-origin "
    "temporal validation with DAPSTOM. Unobserved pairs are treated as unlabelled rather than "
    "confirmed absences, and predictions are evaluated using prevalence-aware ranking, "
    "calibration and fixed-effort metrics. [RESULT REQUIRED: insert cross-network, "
    "ecosystem-transfer and temporal-validation effect sizes with 95% confidence intervals.] "
    "These analyses will test whether local directed structure and biological attributes encode "
    "trophic assembly rules that transfer among ecosystems. [PLANNED—NOT YET IMPLEMENTED: the "
    "final ecological conclusion will be written only after all evidence gates are passed.]"
)

INTRODUCTION = [
    (
        "Food webs describe who consumes whom and provide a compact representation of the routes "
        "through which energy, biomass and perturbations move through ecological communities. "
        "Their topology is associated with persistence, stability and responses to environmental "
        "change, but every empirical web is an incomplete record of a much larger set of possible "
        "interactions. Sampling effort, taxonomic resolution, life stage, season and local "
        "abundance all affect whether an interaction is observed. Consequently, an absent record "
        "cannot generally be interpreted as evidence that an interaction is biologically "
        "impossible. Missing links can alter estimates of connectance, trophic position and "
        "network robustness, and can obscure the species or interactions most likely to transmit "
        "future change. {CITE:WILLIAMS2000} {CITE:DUNNE2002} {CITE:GRAY2015} {CITE:TERRY2020}"
    ),
    (
        "Link prediction offers a way to prioritize plausible but unrecorded interactions. "
        "Classical approaches use local similarity, hierarchical structure, species traits or "
        "ensembles of predictors, whereas recent ecological studies frame interaction inference "
        "as a supervised or recommender-system problem. {CITE:CLAUSET2008} {CITE:LU2011} "
        "{CITE:DESJARDINS2017} {CITE:GHASEMIAN2020} However, evaluation choices can dominate the "
        "apparent performance of a model. Treating all unobserved pairs as true negatives creates "
        "label noise; restricting negatives to easily rejected pairs can inflate discrimination; "
        "and ROC area can remain optimistic when plausible links are rare. Ecological link "
        "prediction therefore requires an explicit candidate universe, prevalence-aware metrics, "
        "calibrated probabilities and tests that reflect the observational process. "
        "{CITE:POISOT2023} {CITE:DAVIS2006} {CITE:SAITO2015} {CITE:KAMPE2026}"
    ),
    (
        "Subgraph-based graph neural networks provide a complementary route to interaction "
        "prediction. SEAL represents each candidate link by its enclosing subgraph and learns "
        "structural patterns rather than relying on a fixed set of hand-crafted scores. "
        "{CITE:SEAL2018} Yet trophic networks are directed: a resource→consumer edge is not "
        "interchangeable with its reverse. Biological attributes such as body size, metabolic "
        "type and life stage can further constrain feasible interactions, but their coverage and "
        "meaning vary among ecosystems. {CITE:BARNES2010} {CITE:BROSE2019} {CITE:POTAPOV2019} "
        "Recent studies have demonstrated cross-community ecological link prediction and have "
        "combined structure and traits on the same collection of 290 food webs. "
        "{CITE:BITON2025} {CITE:VANKLEUNEN2026} These advances make within-web accuracy alone an "
        "insufficient test of whether a model has learned transferable trophic assembly rules."
    ),
    (
        "Here we ask whether direction-aware subgraph learning can transfer to food webs and "
        "ecosystems not represented during training while retaining useful uncertainty estimates "
        "under incomplete observation. We separate the contributions of candidate orientation, "
        "directed message passing, node attributes and contextual edge covariates through matched "
        "ablations. We then propose network-disjoint cross-validation, leave-one-ecosystem-out "
        "evaluation, controlled missingness and rolling-origin prediction in the independent "
        "DAPSTOM record. [PLANNED—NOT YET IMPLEMENTED: the cross-network model, positive–unlabelled "
        "objective, calibration analysis and DAPSTOM evaluation remain evidence gates.] We "
        "hypothesize that direction will provide the most consistent structural gain, that traits "
        "will be most informative in sparse but well-characterized webs, and that calibrated "
        "positive–unlabelled learning will improve the reliability of ranked candidate links. "
        "These hypotheses will be retained, revised or rejected only after evaluation on frozen, "
        "shared splits."
    ),
]


RESULTS = [
    (
        "Transductive baselines and matched ablations",
        "[IMPLEMENTED—VERIFY PROVENANCE] GPU-enabled ten-run cohorts exist for the historical "
        "SEAL implementation and for the node-attributed directed-candidate implementation. The "
        "latter preserves candidate order and directional masking but performs message passing on "
        "a weak undirected projection. These cohorts are pilot baselines, not evidence of transfer. "
        "[RESULT REQUIRED: after the four-variant weighted campaign returns VALIDATION_OK, report "
        "paired per-web differences in PR-AUC, ROC-AUC, MCC and calibration, with 95% network-level "
        "bootstrap intervals and separate summaries for negative-pool fallback runs.]"
    ),
    (
        "Transfer to unseen food webs",
        "[PLANNED—NOT YET IMPLEMENTED] Train every model on identical network-disjoint folds and "
        "evaluate complete held-out food webs. [RESULT REQUIRED: report average precision relative "
        "to prevalence, Precision@K, Recall@K, Brier score and expected calibration error; give the "
        "number of held-out networks and uncertainty across networks, not across individual "
        "candidate edges.]"
    ),
    (
        "Transfer across ecosystems and missingness regimes",
        "[PLANNED—NOT YET IMPLEMENTED] Evaluate leave-one-ecosystem-out transfer for lakes, streams, "
        "marine, terrestrial aboveground and terrestrial belowground webs, followed by controlled "
        "10%, 30% and 50% link removal under random and bias-aware observation mechanisms. "
        "[RESULT REQUIRED: identify which model components generalize consistently and relate "
        "heterogeneity to connectance, web size, ecosystem and attribute coverage.]"
    ),
    (
        "Temporal validation with DAPSTOM",
        "[PLANNED—NOT YET IMPLEMENTED] Use rolling-origin splits so that records available through "
        "time t predict interactions first supported at t+1. [RESULT REQUIRED: distinguish "
        "warm-start from cold-start taxa; report fixed-effort recovery and calibration, data "
        "coverage, exclusions and sampling-effort sensitivity.]"
    ),
    (
        "Ecological interpretation and failure modes",
        "[RESULT REQUIRED] Quantify when structural motifs, direction, trait families and edge "
        "covariates improve or degrade predictions. Report false-positive and false-negative "
        "patterns, abstention behaviour, attribute missingness and node-level candidate priorities. "
        "No claim that predicted links reconstruct a complete food web will be made unless the "
        "eligible candidate universe and its scored coverage are explicitly documented."
    ),
]


DISCUSSION = [
    (
        "[RESULT REQUIRED] Open the Discussion with one evidence-backed sentence stating the "
        "principal cross-network result and its uncertainty, followed by the broad ecological "
        "interpretation. The sentence must identify the inferential unit as the food web and must "
        "not infer transfer from within-web edge splits."
    ),
    (
        "The interpretation will separate predictive utility from evidence that an interaction "
        "occurs in nature. Unobserved trophic links can reflect true absence, incomplete sampling, "
        "taxonomic aggregation or changing environmental opportunity. Predictions should "
        "therefore be presented as ranked hypotheses whose reliability depends on the candidate "
        "set and observation process, not as replacements for empirical validation. "
        "{CITE:KAMPE2026} {CITE:HABEDANK2025}"
    ),
    (
        "[RESULT REQUIRED] Discuss whether the added complexity of node attributes, scalar edge "
        "gates or edge-conditioned filters is justified by transfer, calibration and robustness, "
        "rather than by a small transductive gain. Relate any ecosystem-specific differences to "
        "food-web rewiring and trait constraints. {CITE:BARTLEY2019} {CITE:CARON2024}"
    ),
    (
        "[PLANNED—NOT YET IMPLEMENTED] The final paragraph will state the scope of generalization, "
        "the most important remaining bias and the practical value of the released splits, model "
        "weights and calibrated candidate rankings. The Nature Ecology & Evolution framing will "
        "be retained only if the completed study demonstrates transfer to unseen networks and "
        "ecosystems, independent temporal validation and a broad ecological conclusion."
    ),
]


METHODS = [
    (
        "Study design and datasets",
        [
            (
                "[IMPLEMENTED—VERIFY PROVENANCE] The development corpus comprises 290 directed "
                "food webs derived from the GATEWAy FoodWebDataBase snapshot dated 10 December "
                "2018. The repository-level inventory contains 18,718 nodes and 220,880 recorded "
                "resource→consumer links. Webs are assigned to marine (n=134), lake (n=55), "
                "terrestrial belowground (n=52), stream (n=28) and terrestrial aboveground (n=21) "
                "ecosystems. The raw interaction compilation and the conversion scripts will be "
                "hashed in the final provenance table. {CITE:GRAY2015} [VERIFY DATASET CITATION: "
                "confirm the preferred persistent citation and redistribution licence for the "
                "GATEWAy v1.0 snapshot.]"
            ),
            (
                "[PLANNED—NOT YET IMPLEMENTED] Independent temporal evaluation will use DAPSTOM, "
                "an integrated database of fish stomach records. Records will be converted to "
                "directed resource→consumer evidence with explicit spatial, temporal, taxonomic "
                "and sampling-effort fields. Only evidence available before each prediction origin "
                "will enter training. [VERIFY DATASET CITATION: confirm the authoritative DAPSTOM "
                "version, access route, licence and citation before submission.]"
            ),
        ],
    ),
    (
        "Food-web representation and provenance",
        [
            (
                "Each food web is represented by a binary adjacency matrix A with Aij=1 when node "
                "i is a resource consumed by node j. Candidate links are therefore ordered pairs. "
                "Self-pairs are excluded from candidate generation. For a held-out positive link, "
                "the resource→consumer edge is removed before subgraph extraction; an independently "
                "observed reverse edge is retained as context. Every derived object will record the "
                "source hash, node-order hash, split hash, configuration hash, software commit and "
                "random seed. {CITE:GARLASCHELLI2004}"
            ),
            (
                "The unit of biological inference is the food web. Candidate edges from the same "
                "web are not treated as independent replicates in uncertainty estimates. Webs "
                "with data-quality failures will be excluded only by predeclared checks, and all "
                "exclusions will be reported with reasons."
            ),
        ],
    ),
    (
        "Node attributes and missingness",
        [
            (
                "[IMPLEMENTED—VERIFY PROVENANCE] The directed pilot augments structural labels with "
                "44 node features: standardized log10 body mass and a mass-missingness indicator; "
                "multi-hot or one-hot encodings of taxonomic resolution, metabolic type and "
                "movement type; and life-stage categories plus a life-stage missingness indicator. "
                "Body mass for a taxon is taken from the MAT record when valid or from the median "
                "positive GATEWAy endpoint mass for that taxon within the web. {CITE:BARNES2010} "
                "{CITE:BROSE2019} {CITE:POTAPOV2019}"
            ),
            (
                "The current attributed MAT builder fits a mass scaler across the complete 290-web "
                "corpus and is therefore restricted to the transductive pilot. "
                "[PLANNED—NOT YET IMPLEMENTED] For network-disjoint evaluation, all imputers, "
                "scalers and vocabularies will be fitted on training networks only. Missing values "
                "will be imputed to the training mean after standardization and accompanied by "
                "explicit missingness indicators; unseen categories will map to an unknown channel."
            ),
        ],
    ),
    (
        "Original and directed SEAL baselines",
        [
            (
                "SEAL converts link prediction into graph classification by extracting an enclosing "
                "subgraph around each candidate and assigning double-radius node labels that encode "
                "distance to its endpoints. {CITE:SEAL2018} The frozen historical implementation "
                "uses one-hop subgraphs, DGCNN with SortPooling, batch size 50, up to 100,000 "
                "training examples, 50 epochs and a fixed score threshold greater than 0.5. It "
                "uses no ecological attributes or node2vec embeddings. {CITE:DGCNN2018}"
            ),
            (
                "[IMPLEMENTED—VERIFY PROVENANCE] The existing SEAL_directed pilot uses ordered "
                "resource→consumer candidates, directional masking, distinct endpoint labels and the "
                "44 node attributes. Its bundled DGCNN backend nevertheless constructs message-"
                "passing matrices from a weak undirected projection. Direction is consequently "
                "retained through endpoint order and labels, not through directed propagation. The "
                "validated ten-run GPU cohorts use 90/10 positive edge splits, one sampled negative "
                "per positive, one-hop subgraphs, 50 epochs and no Tukey or post-hoc outlier filter."
            ),
        ],
    ),
    (
        "Contextual edge covariates and weighted variants",
        [
            (
                "[IMPLEMENTED—VERIFY PROVENANCE] SEAL-directed-weighted adds a 17-channel vector to "
                "each observed context edge: log10 consumer/resource mass ratio, two endpoint-mass "
                "missingness indicators, eight interaction-type channels, three interaction-"
                "dimensionality channels and three interaction-classification channels. These are "
                "contextual edge covariates, not measurements of diet proportion, biomass flux, "
                "interaction frequency or interaction strength. Duplicate records are aggregated "
                "deterministically; categorical conflicts remain multi-hot."
            ),
            (
                "The target edge and its metadata are removed before candidate classification. "
                "Four paired variants share each frozen split: a node-attributed weak-projection "
                "baseline; directed message passing without edge attributes; directed messages "
                "scaled by a learned scalar gate; and direction-specific edge-conditioned filters. "
                "{CITE:ECC2017} [RESULT REQUIRED: the 11,600-run campaign must return VALIDATION_OK "
                "before any comparative result or runtime claim is inserted.]"
            ),
        ],
    ),
    (
        "Fully directed transferable model",
        [
            (
                "[PLANNED—NOT YET IMPLEMENTED] The primary cross-network model will be a separate "
                "PyTorch Geometric implementation rather than a reinterpretation of the legacy "
                "DGCNN code. {CITE:PYG2019} It will use independent incoming, outgoing and self "
                "transformations, preserve resource and consumer endpoint identities, batch "
                "subgraphs from multiple food webs and support structure-only and trait-aware "
                "ablations. {CITE:GCN2017} {CITE:GRAPHSAGE2017}"
            ),
            (
                "Parity tests will first require the new undirected mode to reproduce the frozen "
                "baseline on identical serialized subgraphs. Reversing a directed edge must change "
                "the directed representation, while the candidate edge itself must be absent from "
                "all structural and attribute channels."
            ),
        ],
    ),
    (
        "Candidate links and positive–unlabelled learning",
        [
            (
                "[IMPLEMENTED—VERIFY PROVENANCE] In the transductive pilot, negative candidates are "
                "directed absent pairs sampled without replacement at a 1:1 ratio. The preferred "
                "pool requires source nodes capable of acting as resources and target nodes capable "
                "of acting as consumers. If this pool is insufficient, sampling falls back to all "
                "directed non-links. Fallback retains the requested class count but changes negative "
                "difficulty; results will therefore be stratified by fallback status."
            ),
            (
                "[PLANNED—NOT YET IMPLEMENTED] In the primary analysis, observed interactions will "
                "be labelled positive and other eligible pairs unlabelled rather than negative. "
                "Training will use a positive–unlabelled objective fitted only on the training "
                "partition. {CITE:ELKAN2008} {CITE:KIRYO2017} Role filtering will be evaluated as a "
                "candidate-set sensitivity analysis, not treated as ground-truth absence."
            ),
        ],
    ),
    (
        "Network-disjoint and ecosystem-transfer evaluation",
        [
            (
                "[PLANNED—NOT YET IMPLEMENTED] The primary evaluation will use five folds of whole "
                "food webs stratified by ecosystem, node count and connectance. No edge, fitted "
                "feature transformation or calibration example from a test web may enter training. "
                "A second protocol will leave out each of the five ecosystem groups in turn. The "
                "published stacked model and scalable structural baselines will consume the same "
                "split manifests and candidate sets. {CITE:BITON2025} {CITE:VANKLEUNEN2026} "
                "{CITE:BUDDY2023} {CITE:NCNC2023}"
            ),
            (
                "The within-web 90/10 experiment will remain a secondary transductive benchmark. "
                "Controlled missingness will remove 10%, 30% and 50% of observed training links "
                "under random, degree-biased, trait-biased and, where possible, sampling-effort-"
                "biased mechanisms. Split files will be serialized once and reused byte-for-byte "
                "across variants."
            ),
        ],
    ),
    (
        "DAPSTOM temporal validation",
        [
            (
                "[PLANNED—NOT YET IMPLEMENTED] DAPSTOM evaluation will use rolling temporal origins "
                "within prespecified spatial groups. At origin t, models will use only records and "
                "taxonomic information available by t and will rank links first supported after t. "
                "Unrecorded pairs will remain unlabelled. Warm-start predictions for taxa previously "
                "seen in the group will form the primary analysis; cold-start taxa will be reported "
                "separately. Sampling effort, taxonomic merging, zero-versus-missing semantics and "
                "data exclusions must be frozen before model evaluation."
            ),
        ],
    ),
    (
        "Metrics, calibration and statistical analysis",
        [
            (
                "Average precision and average precision relative to candidate prevalence will be "
                "primary ranking metrics. Precision@K and Recall@K will be reported for fixed link "
                "and sampling-effort budgets. ROC-AUC, F1, MCC, precision, recall and true skill "
                "statistic will be secondary. {CITE:DAVIS2006} {CITE:SAITO2015} {CITE:ALLOUCHE2006}"
            ),
            (
                "Probabilities will be calibrated using validation data only, with Brier score, "
                "expected calibration error and reliability diagrams reported on held-out webs. "
                "{CITE:GUO2017} Optional abstention will be summarized by coverage–risk curves. "
                "Conformal false-discovery control will be considered only if its exchangeability "
                "assumptions are defensible for the frozen ecological splits. {CITE:DU2026}"
            ),
            (
                "Seed-level metrics will first be averaged within each food web. Contrasts between "
                "models will then be paired by web and split. Principal estimates will report the "
                "median, interquartile range, mean paired difference and 95% confidence interval from "
                "10,000 network-clustered bootstrap replicates, stratified by ecosystem. Secondary "
                "hypothesis tests, if retained, will use Holm correction. No Tukey filtering or "
                "post-hoc outlier deletion will be applied without a predeclared data-quality rule "
                "and a complete sensitivity analysis."
            ),
        ],
    ),
    (
        "Reproducibility, software and computing resources",
        [
            (
                "Each run will preserve the resolved configuration, source commit, dependency "
                "versions, random seed, input and split hashes, metrics, runtime, terminal log and "
                "success marker. Pilot campaigns used CUDA-enabled PyTorch on the Queen Mary "
                "University of London Apocrita facility. [IMPLEMENTED—VERIFY PROVENANCE: insert the "
                "frozen commits, configuration hashes, Slurm job identifiers and validator summaries "
                "for every result retained in the paper.]"
            ),
            (
                "A clean environment must reproduce at least one principal figure from source data "
                "and manifests. Code, immutable splits, derived results, final weights and figure "
                "scripts will be archived under a versioned release, subject to source-data licences."
            ),
        ],
    ),
    (
        "AI-assisted writing disclosure",
        [
            (
                "OpenAI Codex was used to assist with manuscript organization, initial English "
                "drafting, language editing and automated preservation checks for the Word OOXML "
                "package. It was not used to generate or alter primary observations or analytical "
                "results. The authors remain responsible for the scientific content, independently "
                "verified all claims and references, and will review the complete manuscript before "
                "submission. [AUTHOR CONFIRMATION REQUIRED: update this disclosure if subsequent AI "
                "use changes.]"
            ),
        ],
    ),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def marker_format(text: str) -> tuple[str | None, str | None, bool]:
    if text.startswith("{CITE:"):
        return "turquoise", "0070C0", True
    if text.startswith("[IMPLEMENTED"):
        return "brightGreen", "006100", True
    if text.startswith("[RESULT REQUIRED"):
        return "pink", "9C0006", True
    return "yellow", "9C6500", True


def run_xml(text: str, *, bold: bool = False) -> str:
    highlight, colour, marker_bold = marker_format(text) if MARKER_RE.fullmatch(text) else (None, None, False)
    properties = []
    if bold or marker_bold:
        properties.append("<w:b/><w:bCs/>")
    if highlight:
        properties.append(f'<w:highlight w:val="{highlight}"/>')
    if colour:
        properties.append(f'<w:color w:val="{colour}"/>')
    rpr = f"<w:rPr>{''.join(properties)}</w:rPr>" if properties else ""
    preserve = ' xml:space="preserve"' if text[:1].isspace() or text[-1:].isspace() else ""
    return f"<w:r>{rpr}<w:t{preserve}>{escape(text)}</w:t></w:r>"


def text_runs(text: str, *, bold: bool = False) -> str:
    parts = MARKER_RE.split(text)
    return "".join(run_xml(part, bold=bold and not MARKER_RE.fullmatch(part or "")) for part in parts if part)


def paragraph_xml(
    text: str,
    *,
    style: str | None = None,
    align: str | None = None,
    bold: bool = False,
    reference: bool = False,
    keep_next: bool = False,
) -> str:
    props = []
    if style:
        props.append(f'<w:pStyle w:val="{style}"/>')
    if keep_next:
        props.append("<w:keepNext/><w:keepLines/>")
    if align:
        props.append(f'<w:jc w:val="{align}"/>')
    if reference:
        props.extend(
            [
                '<w:ind w:left="360" w:hanging="360"/>',
                '<w:spacing w:after="80" w:line="240" w:lineRule="auto"/>',
                '<w:jc w:val="left"/>',
            ]
        )
    ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
    return f"<w:p>{ppr}{text_runs(text, bold=bold)}</w:p>"


def word_count(text: str) -> int:
    clean = MARKER_RE.sub("", text)
    return len(re.findall(r"\b[\w–—'-]+\b", clean, flags=re.UNICODE))


def load_references(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not 35 <= len(rows) <= 45:
        raise ValueError(f"Expected 35–45 working references, found {len(rows)}")
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Citation ledger contains duplicate keys")
    for row in rows:
        if not row["formatted_reference"] or not row["doi_or_primary_url"]:
            raise ValueError(f"Incomplete reference metadata for {row['key']}")
    return rows


def manuscript_paragraphs(references: list[dict[str, str]]) -> list[str]:
    paragraphs = [
        paragraph_xml(
            "Transferable direction-aware subgraph learning for trophic interaction prediction across ecosystems",
            style="Title",
            align="center",
            keep_next=True,
        ),
        paragraph_xml(
            "Working manuscript—provisional title and claims remain result dependent",
            style="Subtitle",
            align="center",
            keep_next=True,
        ),
        paragraph_xml(
            "Jorge Eduardo Castro Cruces¹, Lachlan G. Burton¹ and Athen Ma¹*",
            align="center",
            bold=True,
        ),
        paragraph_xml(
            "¹School of Electronic Engineering and Computer Science, Queen Mary University of London, London, UK",
            align="center",
        ),
        paragraph_xml(
            "*Correspondence and requests for materials should be addressed to A.M. "
            "[CORRESPONDING AUTHOR EMAIL REQUIRED]",
            align="center",
        ),
        paragraph_xml(
            "Keywords: food webs; trophic interactions; ecological networks; graph neural "
            "networks; link prediction; cross-ecosystem transfer; positive–unlabelled learning; "
            "uncertainty calibration",
            bold=False,
        ),
        paragraph_xml(
            "Editorial status key: [IMPLEMENTED—VERIFY PROVENANCE] identifies existing code or "
            "outputs that still require a frozen evidence link; [PLANNED—NOT YET IMPLEMENTED] "
            "identifies proposed work; [RESULT REQUIRED] identifies claims or values that cannot "
            "be written before the analyses are complete. {CITE:KEY} markers will be replaced by "
            "Mendeley-generated Nature citations in the finalization pass.",
            bold=True,
        ),
        paragraph_xml("Abstract", style="Heading1", keep_next=True),
        paragraph_xml(ABSTRACT),
    ]

    # Nature Ecology & Evolution Articles use an unheaded introduction.
    paragraphs.extend(paragraph_xml(text) for text in INTRODUCTION)

    paragraphs.append(paragraph_xml("Results", style="Heading1", keep_next=True))
    for heading, text in RESULTS:
        paragraphs.append(paragraph_xml(heading, style="Heading2", keep_next=True))
        paragraphs.append(paragraph_xml(text))

    paragraphs.append(paragraph_xml("Discussion", style="Heading1", keep_next=True))
    paragraphs.extend(paragraph_xml(text) for text in DISCUSSION)

    paragraphs.append(paragraph_xml("Methods", style="Heading1", keep_next=True))
    for heading, texts in METHODS:
        paragraphs.append(paragraph_xml(heading, style="Heading2", keep_next=True))
        paragraphs.extend(paragraph_xml(text) for text in texts)

    paragraphs.extend(
        [
            paragraph_xml("Data availability", style="Heading1", keep_next=True),
            paragraph_xml(
                "The GATEWAy-derived input snapshot used for method development is retained with "
                "source and derived-data hashes in the project archive. [VERIFY DATASET CITATION: "
                "insert the authoritative GATEWAy access URL, licence and persistent identifier.] "
                "[PLANNED—NOT YET IMPLEMENTED] DAPSTOM-derived evidence and immutable temporal split "
                "manifests will be deposited or linked according to CEFAS licensing conditions. "
                "[DOI/REPOSITORY PENDING: identify the repository containing the minimum dataset, "
                "split manifests and source data for each main and Extended Data figure.]"
            ),
            paragraph_xml("Code availability", style="Heading1", keep_next=True),
            paragraph_xml(
                "Custom code is currently developed in the qmul-phd-framework repository. "
                "[DOI/REPOSITORY PENDING: create a public, versioned release containing source code, "
                "pinned environments, resolved configurations, model weights, validators and "
                "figure-generation scripts. Replace this sentence with the archival DOI and review "
                "access instructions before submission.]"
            ),
            paragraph_xml("References", style="Heading1", keep_next=True),
            paragraph_xml(
                "Working reference set. Highlighted citation keys are stable drafting identifiers "
                "and must be replaced with Mendeley-generated Nature numbering before submission."
            ),
        ]
    )
    for row in references:
        paragraphs.append(
            paragraph_xml(
                f"{{CITE:{row['key']}}} {row['formatted_reference']} {row['doi_or_primary_url']}",
                reference=True,
            )
        )

    paragraphs.extend(
        [
            paragraph_xml("Acknowledgements", style="Heading1", keep_next=True),
            paragraph_xml(
                "This research utilised Queen Mary University of London's Apocrita HPC facility, "
                "supported by QMUL Research-IT (https://doi.org/10.5281/zenodo.438045). A.M. "
                "acknowledges research funding from the Natural Environment Research Council "
                "(NE/Y001184/1). [AUTHOR CONFIRMATION REQUIRED: verify funder wording, grant holder "
                "and any additional funding or data-provider acknowledgements.]"
            ),
            paragraph_xml("Author contributions", style="Heading1", keep_next=True),
            paragraph_xml(
                "J.E.C.C.: Conceptualization, Methodology, Software, Data curation, Formal analysis, "
                "Investigation, Validation, Visualization, Project administration and Writing—"
                "original draft. L.G.B.: Methodology, Validation and Writing—review & editing. A.M.: "
                "Conceptualization, Supervision, Funding acquisition, Resources, Project "
                "administration and Writing—review & editing. [AUTHOR CONFIRMATION REQUIRED: all "
                "authors must approve the final CRediT statement and author order.]"
            ),
            paragraph_xml("Competing interests", style="Heading1", keep_next=True),
            paragraph_xml("The authors declare no competing interests."),
            paragraph_xml(
                "Supplementary information and reporting summary",
                style="Heading1",
                keep_next=True,
            ),
            paragraph_xml(
                "[REPORTING SUMMARY REQUIRED] Complete the Nature Portfolio Reporting Summary; "
                "provide source data for all plotted values; include split, leakage and exclusion "
                "audits; and limit the final Article to no more than six main display items and ten "
                "Extended Data items. Proposed main displays are: method overview; matched ablation; "
                "network and ecosystem transfer; robustness to missingness; ecological "
                "interpretation; and DAPSTOM temporal validation."
            ),
        ]
    )
    return paragraphs


PARAGRAPH_RE = re.compile(rb"<w:p(?:\s[^>]*)?>.*?</w:p>", flags=re.DOTALL)


def document_body(xml: bytes) -> tuple[int, int, bytes]:
    body_open = xml.find(b"<w:body>")
    body_close = xml.rfind(b"</w:body>")
    if body_open < 0 or body_close < 0:
        raise ValueError("Could not locate w:body in template document.xml")
    content_start = body_open + len(b"<w:body>")
    return content_start, body_close, xml[content_start:body_close]


def acknowledgement_tail(xml: bytes) -> bytes:
    _, _, body = document_body(xml)
    paragraphs = list(PARAGRAPH_RE.finditer(body))
    for paragraph in paragraphs:
        text = b"".join(
            re.findall(rb"<w:t(?:\s[^>]*)?>(.*?)</w:t>", paragraph.group(), re.DOTALL)
        )
        if text == b"Acknowledgements":
            return body[paragraph.start() :]
    raise ValueError("Could not locate the original Acknowledgements boundary")


def paragraph_identity(seed: str) -> tuple[bytes, bytes]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()

    def word_identifier(chunk: bytes) -> bytes:
        # Word-generated w14 paragraph/text identifiers use the positive
        # 31-bit range.  XML parsers accept the full unsigned 32-bit range,
        # but Word treats values with the high bit set as repairable content.
        value = int.from_bytes(chunk, byteorder="big") & 0x7FFFFFFF
        if value == 0:
            value = 1
        return f"{value:08X}".encode("ascii")

    return word_identifier(digest[:4]), word_identifier(digest[4:8])


def paragraph_from_template(raw: bytes, text: str, *, seed: str) -> bytes:
    """Clone an original paragraph's formatting and replace only its text."""

    opening_end = raw.find(b">")
    if opening_end < 0:
        raise ValueError("Invalid paragraph template")
    opening = raw[: opening_end + 1]
    para_id, text_id = paragraph_identity(seed)
    opening = re.sub(rb'w14:paraId="[^"]*"', b'w14:paraId="' + para_id + b'"', opening)
    opening = re.sub(rb'w14:textId="[^"]*"', b'w14:textId="' + text_id + b'"', opening)

    ppr_match = re.search(rb"<w:pPr>.*?</w:pPr>", raw, flags=re.DOTALL)
    ppr = ppr_match.group() if ppr_match else b""
    run_match = re.search(rb"<w:r(?:\s[^>]*)?>.*?</w:r>", raw, flags=re.DOTALL)
    rpr = b""
    if run_match:
        rpr_match = re.search(rb"<w:rPr>.*?</w:rPr>", run_match.group(), flags=re.DOTALL)
        if rpr_match:
            rpr = rpr_match.group()
    encoded = escape(text).encode("utf-8")
    run = b'<w:r>' + rpr + b'<w:t xml:space="preserve">' + encoded + b"</w:t></w:r>"
    return opening + ppr + run + b"</w:p>"


def build_scaffold_paragraphs(
    template_xml: bytes,
    references: list[dict[str, str]],
) -> list[bytes]:
    """Return manuscript paragraphs using only formatting from the scaffold."""

    _, _, body = document_body(template_xml)
    source = [match.group() for match in PARAGRAPH_RE.finditer(body)]
    if len(source) != 44:
        raise ValueError(f"Expected 44 paragraphs in the original scaffold, found {len(source)}")

    title_template = source[0]
    normal_template = source[34]
    heading1_template = source[14]
    heading2_template = source[19]

    output: list[bytes] = []
    add_count = 0

    def add(template: bytes, text: str) -> None:
        nonlocal add_count
        add_count += 1
        output.append(
            paragraph_from_template(template, text, seed=f"seal-draft-{add_count}-{text}")
        )

    add(
        title_template,
        "Transferable direction-aware subgraph learning for trophic interaction prediction "
        "across ecosystems",
    )
    output.extend([source[1], source[2], source[3]])
    add(source[4], "*Corresponding author [CORRESPONDING AUTHOR EMAIL REQUIRED]")
    output.extend([source[5], source[6], source[7]])
    add(
        source[8],
        "Keywords: food webs; trophic interactions; ecological networks; graph neural networks; "
        "link prediction; cross-ecosystem transfer; positive–unlabelled learning; uncertainty "
        "calibration",
    )
    add(
        normal_template,
        "Editorial status key: [IMPLEMENTED—VERIFY PROVENANCE] identifies existing code or "
        "outputs that still require a frozen evidence link; [PLANNED—NOT YET IMPLEMENTED] "
        "identifies proposed work; [RESULT REQUIRED] identifies claims or values that cannot be "
        "written before the analyses are complete. {CITE:KEY} markers will be replaced by "
        "Mendeley-generated Nature citations in the finalization pass.",
    )
    output.append(source[9])

    add(source[10], "Abstract")
    add(normal_template, ABSTRACT)
    add(source[12], "Introduction")
    for text in INTRODUCTION:
        add(normal_template, text)

    add(source[14], "Results")
    for heading, text in RESULTS:
        add(heading2_template, heading)
        add(normal_template, text)

    add(source[16], "Discussion")
    for text in DISCUSSION:
        add(normal_template, text)

    add(source[18], "Methods")
    for heading, texts in METHODS:
        add(heading2_template, heading)
        for text in texts:
            add(normal_template, text)

    add(heading1_template, "Code availability")
    add(
        normal_template,
        "Custom code is currently developed in the qmul-phd-framework repository. "
        "[DOI/REPOSITORY PENDING: create a public, versioned release containing source code, "
        "pinned environments, resolved configurations, model weights, validators and "
        "figure-generation scripts. Replace this sentence with the archival DOI and review "
        "access instructions before submission.]",
    )

    add(source[31], "References")
    add(
        normal_template,
        "Working reference set. Citation keys are stable drafting identifiers and must be "
        "replaced with Mendeley-generated Nature numbering before submission.",
    )
    for row in references:
        add(
            normal_template,
            f"{{CITE:{row['key']}}} {row['formatted_reference']} {row['doi_or_primary_url']}",
        )
    return output


def replace_document_body(template_xml: bytes, references: list[dict[str, str]]) -> bytes:
    content_start, body_close, _ = document_body(template_xml)
    new_prefix = b"".join(build_scaffold_paragraphs(template_xml, references))
    unchanged_tail = acknowledgement_tail(template_xml)
    output = template_xml[:content_start] + new_prefix + unchanged_tail + template_xml[body_close:]
    ET.fromstring(output)
    return output


def validate_docx(
    path: Path,
    template_parts: dict[str, bytes],
    references: list[dict[str, str]],
) -> dict[str, object]:
    with zipfile.ZipFile(path) as package:
        bad = package.testzip()
        if bad:
            raise ValueError(f"Corrupt ZIP member: {bad}")
        names = package.namelist()
        if len(names) != len(set(names)):
            raise ValueError("DOCX contains duplicate ZIP members")
        output_parts = {name: package.read(name) for name in names}
        for name, data in output_parts.items():
            if name.endswith((".xml", ".rels")):
                ET.fromstring(data)

    if set(output_parts) != set(template_parts):
        raise ValueError("DOCX package member set changed")
    changed = [name for name in template_parts if template_parts[name] != output_parts[name]]
    if changed != [DOCUMENT_XML]:
        raise ValueError(f"Unexpected package changes: {changed}")

    root = ET.fromstring(output_parts[DOCUMENT_XML])
    ns = {"w": W_NS}
    paragraphs = root.findall(".//w:body/w:p", ns)
    texts = ["".join(node.text or "" for node in p.findall(".//w:t", ns)) for p in paragraphs]
    full_text = "\n".join(texts)
    paragraph_ids = [
        int(value, 16)
        for value in re.findall(
            rb'w14:paraId="([0-9A-Fa-f]{8})"', output_parts[DOCUMENT_XML]
        )
    ]
    if not paragraph_ids or any(not 0 < value <= 0x7FFFFFFF for value in paragraph_ids):
        raise ValueError("Word paragraph identifiers are outside the valid positive 31-bit range")
    if len(paragraph_ids) != len(set(paragraph_ids)):
        raise ValueError("Word paragraph identifiers are not unique")
    required = [
        "Abstract",
        "Introduction",
        "Results",
        "Discussion",
        "Methods",
        "Code availability",
        "References",
        "Acknowledgements",
        "Author contributions",
        "Competing interests",
        "Data availability",
    ]
    missing = [value for value in required if value not in texts]
    if missing:
        raise ValueError(f"Missing required headings: {missing}")
    if full_text.count("{CITE:") < len(references):
        raise ValueError("Citation markers were lost")

    source_tail = acknowledgement_tail(template_parts[DOCUMENT_XML])
    output_tail = acknowledgement_tail(output_parts[DOCUMENT_XML])
    if output_tail != source_tail:
        raise ValueError("Acknowledgements and subsequent content changed")
    if b"<w:highlight" in output_parts[DOCUMENT_XML] or b"<w:shd" in output_parts[DOCUMENT_XML]:
        raise ValueError("The rebuilt manuscript introduced highlighting or shading")

    webextensions = sorted(name for name in output_parts if "webextension" in name.lower())
    if not webextensions:
        raise ValueError("Mendeley webextension parts are missing")

    return {
        "zip_members": len(output_parts),
        "changed_package_parts": changed,
        "unchanged_package_parts": len(output_parts) - len(changed),
        "webextension_parts": webextensions,
        "webextension_sha256": {
            name: sha256_bytes(output_parts[name]) for name in webextensions
        },
        "paragraphs": len(paragraphs),
        "paragraph_identifiers": len(paragraph_ids),
        "paragraph_identifiers_positive_31_bit": True,
        "working_references": len(references),
        "citation_markers": full_text.count("{CITE:"),
        "implemented_markers": full_text.count("[IMPLEMENTED—VERIFY PROVENANCE"),
        "planned_markers": full_text.count("[PLANNED—NOT YET IMPLEMENTED"),
        "result_markers": full_text.count("[RESULT REQUIRED"),
        "formatting_policy": "original scaffold paragraph formatting only",
        "acknowledgements_onward_unchanged": True,
        "acknowledgements_tail_sha256": sha256_bytes(output_tail),
        "introduced_highlights_or_shading": False,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    target = args.docx.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    backup = target.with_name(target.stem + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(target, backup)
    correction_backup = target.with_name(target.stem + CORRECTION_BACKUP_SUFFIX)
    if not correction_backup.exists():
        shutil.copy2(target, correction_backup)
    template = backup

    ledger = args.ledger.expanduser().resolve()
    references = load_references(ledger)
    if word_count(ABSTRACT) > 200:
        raise ValueError(f"Abstract exceeds 200 words: {word_count(ABSTRACT)}")
    introduction_words = sum(word_count(text) for text in INTRODUCTION)
    if not 430 <= introduction_words <= 600:
        raise ValueError(f"Introduction word count outside 430–600: {introduction_words}")
    main_words = introduction_words + sum(word_count(text) for _, text in RESULTS) + sum(
        word_count(text) for text in DISCUSSION
    )
    if main_words > 3500:
        raise ValueError(f"Main text exceeds 3,500 words: {main_words}")

    with zipfile.ZipFile(template) as package:
        infos = package.infolist()
        template_parts = {info.filename: package.read(info.filename) for info in infos}
    output_xml = replace_document_body(
        template_parts[DOCUMENT_XML], references
    )

    fd, temp_name = tempfile.mkstemp(
        prefix=target.stem + ".building.", suffix=".docx", dir=target.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        with zipfile.ZipFile(temp_path, "w") as output:
            for info in infos:
                data = output_xml if info.filename == DOCUMENT_XML else template_parts[info.filename]
                output.writestr(info, data)
        validation = validate_docx(temp_path, template_parts, references)
        os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    audit = {
        "status": "DOCX_VALIDATION_OK",
        "target": str(target),
        "backup": str(backup),
        "pre_correction_backup": str(correction_backup),
        "template_sha256": sha256_file(template),
        "output_sha256": sha256_file(target),
        "ledger": str(ledger),
        "ledger_sha256": sha256_file(ledger),
        "abstract_words_excluding_markers": word_count(ABSTRACT),
        "introduction_words_excluding_markers": introduction_words,
        "main_text_words_excluding_markers": main_words,
        **validation,
    }
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", type=Path, default=DEFAULT_DOCX)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=repo_root / "docs/SEAL/manuscript/citation_ledger.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=repo_root / "docs/SEAL/manuscript/build_audit_2026-10-27.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))
