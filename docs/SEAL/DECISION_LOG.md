# SEAL paper decision log

This is an append-only record of decisions that materially affect scientific
scope, implementation, evaluation, authorship, publication, or reproducibility.
Reversals must add a new entry and link to the decision being superseded.

## Decision template

```markdown
## DNNN -- short title

- **Date:** YYYY-MM-DD
- **Status:** Accepted | Superseded | Rejected
- **Context:** Why a decision was required.
- **Decision:** What was selected.
- **Alternatives:** Material alternatives considered.
- **Rationale:** Evidence and trade-offs.
- **Consequences:** Required work, exclusions, and revisit condition.
```

## D001 -- combine ecological and methodological contribution

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** A model-only benchmark would have limited ecological reach,
  while an application-only paper would not establish sufficient novelty.
- **Decision:** Centre the paper on an ecological transfer question supported
  by a new direction-aware and uncertainty-aware SEAL method.
- **Alternatives:** Pure algorithm paper; application-only food-web paper.
- **Rationale:** This gives the strongest alignment with a high-impact ecology
  journal while maintaining a defensible technical contribution.
- **Consequences:** Every technical experiment must support an ecological
  hypothesis or a necessary validity check.

## D002 -- make robust transfer the principal claim

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** Direction and traits alone are already represented in recent
  ecological link-prediction work.
- **Decision:** Test whether trophic assembly rules transfer to unseen food
  webs and ecosystems under incomplete and biased observations.
- **Alternatives:** Focus only on directed message passing; focus only on
  interpretability.
- **Rationale:** Network-disjoint transfer distinguishes the work from the
  current within-network SEAL results and from the WLNM paper.
- **Consequences:** Network-disjoint cross-validation is primary; transductive
  performance is secondary.

## D003 -- use DAPSTOM for external temporal validation

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** The 290 Gateway food webs are also used in a 2026 Nature
  Communications study, so an independent test is required.
- **Decision:** Use DAPSTOM sea--decade networks in a rolling-origin temporal
  evaluation.
- **Alternatives:** Hengill thermal streams; acquisition of a new external
  dataset.
- **Rationale:** DAPSTOM provides an independent predator--prey source,
  temporal structure, spatial grouping, and sampling-effort information.
- **Consequences:** DAPSTOM preprocessing and licence checks are within scope;
  unobserved pairs must remain unlabelled.

## D004 -- build a new PyTorch Geometric implementation

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** The current DGCNN backend uses weak undirected projections and
  is poorly suited to pooled learning across networks.
- **Decision:** Develop `src/python/eco_seal/` in PyTorch Geometric and retain
  the existing implementation as a frozen baseline.
- **Alternatives:** Extend the legacy backend; perform a gradual migration only
  after additional legacy experiments.
- **Rationale:** PyG supports directed edge handling, cross-network batching,
  modern baselines, GPU execution, and maintainable testing.
- **Consequences:** Parity with the legacy baseline is required before claims
  about the directed architecture.

## D005 -- use a six-month weekly programme

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** The paper must progress alongside the active WLNM manuscript.
- **Decision:** Allocate 15--20 hours per week for 26 weeks, with evidence-based
  gates at Weeks 4, 12, and 20.
- **Alternatives:** Three-to-four-month reduced study; nine-to-twelve-month
  expanded programme.
- **Rationale:** Six months permits implementation, controlled experiments,
  external validation, interpretation, and cumulative writing.
- **Consequences:** New ideas enter the active experiment matrix only through a
  documented decision and must displace equivalent work.

## D006 -- use a conditional Nature publication ladder

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** Journal choice determines the breadth of evidence and narrative.
- **Decision:** Consider Nature Ecology & Evolution only after the Week 20
  evidence gate; otherwise target Nature Communications or Methods in Ecology
  and Evolution according to the type of contribution achieved.
- **Alternatives:** Commit immediately to MEE; commit immediately to Nature
  Communications.
- **Rationale:** The journal must follow the verified strength and breadth of
  the result.
- **Consequences:** No journal-specific performance claim will be written
  before the Week 20 gate.

## D007 -- adopt an open and auditable release

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** Transfer claims require reproducible splits and artefact-level
  provenance.
- **Decision:** Release code, split manifests, resolved configurations, final
  weights, and derived results, subject to data licences.
- **Alternatives:** Code and results only; private artefacts until acceptance.
- **Rationale:** Open manifests and weights improve reproducibility, review,
  uptake, and evidence of project leadership.
- **Consequences:** Provenance and release-readiness are acceptance criteria,
  not tasks deferred until submission.

## D008 -- formalise the verbal lead-authorship agreement

- **Date:** 2026-09-24
- **Status:** Accepted
- **Context:** A verbal agreement exists that the researcher will lead and be
  first author.
- **Decision:** Week 1 will create a provisional CRediT matrix, author-order
  note, and written confirmation of expectations.
- **Alternatives:** Leave authorship implicit until manuscript drafting.
- **Rationale:** Early transparency protects collaborators, makes leadership
  measurable, and reduces later disputes.
- **Consequences:** Contributions and author order will be reviewed monthly and
  updated according to actual work.

