# qmul-phd-framework

## ✅ 1. **Define a Research Workflow Structure**

Structure your work using the **DS/ML research lifecycle**:

```
[Problem Definition] → [Data Preparation] → [Methodology/Algorithm Setup] → [Experimentation] → [Results & Analysis] → [Interpretation] → [Publication]
```

Create folders (or GitHub repos) that mirror this logic. Example:

```
phd-wlnm-analysis/
├── data/
│   ├── raw/
│   └── processed/
├── experiments/
│   ├── small_webs/
│   └── large_webs/
├── figures/
├── notebooks/
├── results/
│   ├── metrics/
│   └── logs/
├── src/
│   └── matlab/
├── docs/
│   ├── progress_logs.md
│   └── report_slides.pptx
├── paper/
│   └── draft.tex
└── README.md
```

---

## ✅ 2. **Log and Visualize Progress the Academic Way**

### Option 1: **Jupyter Notebooks** (Python-centric)

Use Jupyter Notebooks to:

* Import experiment results (CSV)
* Generate ROC, PR, confusion matrix plots
* Log observations and TODOs inline using Markdown
* Export figures directly to `figures/`
* Save as HTML or PDF to share with your supervisor

> If your core code is in MATLAB, you can still generate CSVs from MATLAB and analyze them in Python/Jupyter. This is **a powerful hybrid strategy**.

### Option 2: **LaTeX/Overleaf** (Paper-ready)

Use Overleaf or LaTeX locally to:

* Write a structured document (even if informal at first): `Introduction`, `Methods`, `Results`, `Discussion`
* Insert high-quality figures from `figures/`
* Add tables with AUC, precision, recall, TP/FP/FN counts
* This doubles as your **paper draft** and your **weekly research log**

### Option 3: **PowerPoint Research Log** (For meetings)

Use a 5-slide template for weekly meetings:

1. **What I did this week**
2. **How I did it (key code/graph/results)**
3. **What I learned**
4. **Challenges or open questions**
5. **Next steps**

Store this in `/docs/report_slides.pptx`, updating slides as needed weekly.

---

## ✅ 3. **Track Experiments Like a Researcher**

Use a clear convention:

### A. CSV Logging Format:

Each row = 1 experiment:

```csv
foodweb_name,n_links,K,max_depth,auc,precision,recall,tp,fp,fn,time
Chesapeake,16041,3,5,0.81,0.72,0.76,120,35,37,02:15
```

### B. Markdown Logs (`progress_logs.md`)

```md
### [2025-05-05] WLNM Tests on Chesapeake

- Params: K=3, max_depth=5
- TP: 120, FP: 35, FN: 37
- AUC: 0.81
- Observation: AUC improved with higher K, but runtime increased to 2+ hours.
- Issue: Mass ordering seems to lower recall.
- Next: Compare with Katz similarity baseline.
```

---

## ✅ 4. **GitHub + Overleaf for Publication**

* Use **GitHub** to version control your code *and* results.

  * Push your `/results`, `/notebooks`, and `/paper` folders.
  * Add Jupyter or Markdown README for each experiment.

* Use **Overleaf** to write your paper in LaTeX.

  * Link to plots and tables generated from CSV/Notebook.
  * Use the log file and PowerPoint as your draft base.

---

## ✅ 5. **Use MATLAB Only for Core Graph Computation**

All post-processing (ROC, PR, visual plots, metrics aggregation) should move to **Python** (or even R), because:

* It's easier to script, plot, and automate with `pandas`, `matplotlib`, `scikit-learn`
* It integrates with Jupyter, Overleaf, GitHub, etc.

You can write a simple bridge like:

```matlab
csvwrite('results/Chesapeake_K3_depth5.csv', [auc, recall, precision, tp, fp, fn])
```

Then read it in Python:

```python
import pandas as pd
df = pd.read_csv('results/Chesapeake_K3_depth5.csv')
```

---

## ✅ Final Academic Advice

> Your **goal** is not just to “run code” but to “build a reproducible, interpretable, and publication-ready pipeline.”

### Summary of Tools I Recommend:

| Task             | Tool                                 |
| ---------------- | ------------------------------------ |
| Core WLNM code   | MATLAB                               |
| Data logging     | CSV + Markdown                       |
| Analysis & plots | Python (pandas, matplotlib, sklearn) |
| Reporting        | Jupyter Notebooks + LaTeX (Overleaf) |
| Meetings         | PowerPoint log or exported Notebook  |
| Version Control  | GitHub                               |

---
