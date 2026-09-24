# SEAL paper workspace

This directory is the canonical workspace for the paper on a transferable,
direction-aware SEAL method for ecological food webs.

## Current status

- **Status:** framework agreed; Week 1 not started
- **Planning date:** 24 September 2026
- **Target horizon:** 26 weeks, ending 28 March 2027
- **Weekly allocation:** 15--20 hours
- **Next milestone:** Week 1 paper charter, scope boundary, and written CRediT agreement

## Working research question

Can a fully directed SEAL model learn trophic assembly rules from local
structure and species traits that transfer to unseen food webs and ecosystems,
while remaining calibrated under incomplete and biased observations?

## Canonical documents

- [Master framework](SEAL_PAPER_FRAMEWORK.md): scientific scope, technical
  design, evaluation protocols, publication gates, and 26-week roadmap.
- [Weekly tracker](WEEKLY_TRACKER.md): objectives, evidence, decisions,
  blockers, and hand-off for each weekly sprint.
- [Decision log](DECISION_LOG.md): durable record of scientific and project
  decisions, including their rationale and consequences.
- [Literature matrix](literature/literature_matrix.csv): structured evidence
  table for papers reviewed during the project.
- [Existing preliminary report](reports/SEAL_directed_train90.docx): snapshot
  of the initial directed-SEAL results; it is not the master plan.

## Weekly operating rule

At the beginning of each week:

1. Read the master framework and the previous tracker entry.
2. Select one primary deliverable and no more than three supporting tasks.
3. Record the planned acceptance test before running experiments.

At the end of each week:

1. Link commits, manifests, figures, tables, and manuscript text.
2. Record decisions and failed paths, not only successful results.
3. Write the next week's first concrete action.
4. Update this dashboard if the active milestone or paper scope changed.

## Separation from the WLNM paper

The WLNM paper concerns within-network reconstruction using matrix encoding and
ecological negative sampling. This SEAL paper concerns cross-network inductive
learning, directed message passing, positive--unlabelled treatment, calibrated
uncertainty, and temporal external validation. Headline results and figures
must not be duplicated between the two papers.

