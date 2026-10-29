# Method

The implementation follows `manuscript_v9/main_v5.tex`, the authoritative local
manuscript, and its accompanying method specification.

## Representations

- **Descriptors:** numeric whole-part measurements from STEP geometry.
- **Compact state:** 14 selected measurements rounded to three decimals, paired
  with the text `unlabeled mechanical CAD part`. The same field mapping builds
  model inputs and compact-state probe features.
- **Typed responses:** ten Choice probabilities and 22 Noul scores. The fixed
  question map includes 13 geometry, seven function, and two manufacturing
  propositions. Jev, CLM, base Laya, and adapted Laya share this input protocol.
- **Surface embedding:** 512 area-sampled points with triangle normals and 16
  fixed nearest neighbors. Two EdgeBlocks use widths 32 and 64. Maximum, mean,
  and population-standard-deviation pooling feeds a 192→64→32 projection.
  A class head trains the encoder; validation macro F1 selects its checkpoint.

Encoder widths are configurable and saved with the checkpoint. Checkpoints from
an earlier architecture retain their original dimensions when loaded.

## Classification

Direct Choice uses the largest class probability. Representation readouts use
training-fitted median imputation, standardization, and logistic regression.
Validation macro F1 selects C from `[0.01, 0.1, 1, 10]`; the selected pipeline is
refitted on training plus validation before test evaluation.

Main run configurations exclude identical-file groups with conflicting source
labels from the aligned classification readouts. The duplicate audit supplies
the exclusions, which are saved beside the evaluation outputs. Acquisition,
encoder training, and retrieval retain their configured cohorts.

The experiment catalog includes descriptor, compact-state, surface, Choice,
Noul, geometry-focused Noul, and concatenated fusion inputs. Metrics include
accuracy, macro F1, balanced accuracy, NLL, and multiclass Brier score.

## Adaptation and prompting

Laya adaptation uses source-class Choice supervision, soft cross-entropy, and
group-relative RLCD. Validation selects the checkpoint; a separate training
partition fits the Choice temperature.

`study-prompt` compares zero demonstrations with four labeled training examples
per class under a shared prompt wrapper and geometric-tier questions. Supports
are balanced by class, selected deterministically, and replaced when a query
shares their exact-file or rounded-state identity. Prompted Choice probabilities
feed direct decisions and the registered fusion readouts. Auxiliary tier outputs
are saved separately from the fusion features.

## Retrieval

Training-fitted imputation and standardization precede cosine similarity.
Surface–response fusion combines similarities as `(1-alpha)*surface + alpha*response`.
Validation nDCG selects alpha from `[0, 0.25, 0.5, 0.75, 1]`.

The candidate pool merges each standalone representation’s top ten neighbors
with ten random candidates. Two reviewers grade blinded pairs from 0 to 3;
agreement is retained and disagreements are adjudicated. Class-stratified query
sampling defaults to five validation and ten test queries per class.

Every method ranks the same judged candidates. nDCG uses gain `2**grade - 1`;
precision, recall, MAP, and MRR use grades 2–3 as relevant. MAP and MRR cover the
complete judged ranking.
