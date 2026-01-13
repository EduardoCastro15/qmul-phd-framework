# qmul-phd-framework

This repository contains the **research framework, data, code, and documentation** supporting my PhD work at **Queen Mary University of London**, focused on:

> **Link prediction in ecological food webs**, with emphasis on
> Weisfeiler–Lehman–based methods (WLNM), backbone extraction, and the ecological interpretation of reduced network representations.

The repository is designed to support **reproducible experiments**, **systematic analysis**, and **publication-ready outputs**.

---

## Research Workflow

The project follows a standard data science / machine learning research lifecycle:

```
Problem Definition
        ↓
Data Preparation
        ↓
Methodology & Algorithm Design
        ↓
Experimentation
        ↓
Results & Analysis
        ↓
Ecological Interpretation
        ↓
Manuscript Preparation
```

The folder structure mirrors this workflow to keep research, experiments, and documentation clearly separated.

---

## Repository Structure (High-level)

```
.
├── data/           # Raw and processed datasets (food webs, metadata)
├── src/            # Core algorithmic implementations (MATLAB)
├── results/        # Experiment outputs (CSV, logs)
├── notebooks/      # Analysis and visualization notebooks (Python)
├── docs/           # Documentation, papers, slides, figures
├── paper/          # Manuscript drafts and publication material
└── README.md
```

---

## Data (`/data`)

* **`raw/`**

  * Original datasets (e.g. Hengill food webs, large food-web collections)
* **`processed/`**

  * Cleaned and derived datasets:

    * Food-web metadata (size, ecosystem, trophic properties)
    * Stratified and size-based subsets
    * CSV and MAT representations used in experiments

All data transformations are scripted and reproducible.

---

## Core Algorithms (`/src`)

### MATLAB (`/src/matlab`)

This folder contains the **core WLNM pipeline**, including:

* Directed WLNM implementations
* Negative sampling strategies
* Backbone-based training regimes
* Subgraph expansion and encoding
* Logging and experiment runners

MATLAB is used **only for core graph computation** and algorithm execution.

---

## Experiments and Results

### Results (`/results`)

* CSV files with per-experiment metrics:

  * AUC, Precision, Recall, F1
  * TP / FP / FN counts
  * Runtime and configuration metadata
* Logs generated during large-scale sweeps

Each row corresponds to a **single experimental configuration**.

### Notebooks (`/notebooks`, `/docs/useful_notebooks`)

Python notebooks are used for:

* Aggregating experiment outputs
* Statistical analysis across food webs
* Visualization (ROC, PR curves, boxplots, scatter plots)
* Ecosystem-level and temperature-driven comparisons

This allows a clean separation between **computation (MATLAB)** and **analysis/visualisation (Python)**.

---

## Documentation (`/docs`)

The `docs/` folder contains:

* **Methodology documentation**

  * Draft and evolving descriptions of the WLNM pipeline
  * Supplementary material for publication
* **Figures and plots**

  * Publication-ready visualizations
* **Slides**

  * Weekly or milestone research updates
* **Reference papers**

  * Key literature related to food webs, link prediction, and network backbones
* **Progress notebooks and logs**

  * Internal research tracking (not all content is versioned for size reasons)

---

## Manuscript Preparation (`/paper`)

This folder is used for:

* Paper drafts (LaTeX / Word)
* Figure integration
* Tables generated from experiment CSVs

The goal is to maintain a **direct link between experiments and manuscript figures**.

---

## Experimental Logging Conventions

Experiments are logged in CSV format for reproducibility.
Example schema:

```csv
ExpID,AUC,TimeElapsed,K,TrainRatio,BackboneRatio,Threshold,Precision,Recall,F1Score,TotalLinks,TrainLinks,TestLinks,BackboneTotal,NonBackboneTotal,BackboneTrainLinks,NonBackboneTrainLinks,BackboneTestLinks,NonBackboneTestLinks
1,0.8925,00:00:08,10,60,20,0.10,0.6970,0.9122,0.7902,1593,955,638,278,1315,56,899,222,416
```

Markdown notes are used alongside results to capture observations, issues, and next steps.

---

## Tooling Summary

| Task                        | Tooling                                     |
| --------------------------- | ------------------------------------------- |
| Core algorithms             | MATLAB                                      |
| Data handling & aggregation | CSV + Python (`pandas`)                     |
| Analysis & visualisation    | Python (`matplotlib`, `seaborn`, `sklearn`) |
| Documentation & drafting    | Markdown, Word, LaTeX (Overleaf)            |
| Version control             | Git & GitHub                                |

---

## Notes on Reproducibility

* All experiments are parameterized and logged.
* Figures are generated from CSV outputs rather than manual editing.
* Large intermediate artifacts (e.g. raw notebooks, logs) are intentionally excluded from version control where appropriate.

---

## Status

This repository is **actively developed** as part of an ongoing PhD project.
Structure and content may evolve as experiments and manuscripts progress.
