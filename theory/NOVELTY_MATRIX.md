# Novelty Matrix

The central boundary of the revision is deliberate: voting-margin facts and
classical majority/noise results are not presented as newly invented. The
contribution is the exact connection of fixed-position prototype relabeling,
prototype-level global influence, and conditional random-label reliability for
the same observed kNN neighborhood, together with an auditable implementation.

| Result or object | Classification | What prior work covers | What this revision claims |
|---|---|---|---|
| One-label binary odd-k pivotality | Direct consequence of majority voting | Classical Boolean influence and pivotal-variable theory | A kNN-specific exact implementation and identity for query/prototype incidence |
| Multiclass directional replacement rule | New formulation for this fixed-neighborhood model | Multiclass vote-margin certificates and poisoning robustness | Exact rule under explicit allowed-label set and deterministic tie priority; no claim of new poisoning certification |
| Query-level PRV versus maximum single-prototype influence | Audit representation with an overlapping poisoning objective | Budget-one label poisoning can equal qR1 when baseline queries are all correct and intervention sets agree | The full directional incidence matrix separates changed predictions, signed error changes, and influence concentration; R1 itself is not claimed as a new poisoning objective |
| Independent label-flip probability for one query | Direct Poisson-binomial derivation | Randomized-smoothing and noisy-label literature provide related robustness models | Exact conditional probability for fixed kNN positions and observed labels, with DP and enumeration checks |
| First-order risk derivative at zero flip probabilities | Direct Boolean influence identity | Classical influence/noise sensitivity | Used as a transparent bridge between deterministic influence and conditional risk, not claimed as a new general theorem |
| Batch-query risk variance | New kNN audit formulation | Efron-Stein and bounded-difference tools give generic bounds | Explicit shared-prototype covariance accounting and computable coverage bounds |
| Heterogeneous local label probabilities | Direct Poisson-binomial extension | kNN statistical stability and imperfect-label theory | A distribution-dependent reliability quantity conditioned on the observed neighborhood geometry |
| Data-review ranking | Application / empirical contribution | Influence functions, data valuation, prototype editing | Compare relabeling-specific exact influence with general training-data scores and report failure cases |
| Fixed-neighborhood influence versus kNN Shapley valuation | Explicit objective separation | Jia et al. compute exact coalition-utility Shapley values for kNN data valuation | Directional influence fixes the observed neighborhood and a replacement label; it is a prediction-change audit, not a valuation or coalition utility |
| Fixed-geometry label poisoning | Explicit overlap with prior work | Centurion et al. optimize label flips on fixed geometric kNN data | State the budget-one equivalence condition and distinguish exact dependence/probability auditing from attack optimization |
| Geometry perturbation | Scope extension and limitation study | Nearest-neighbor robustness and feature perturbations | Separate the paper's fixed-label audit from the secondary one-prototype motion check |

## Claims deliberately not made

- No claim that the vote-gap rule is the first robustness certificate for kNN.
- No claim that PRV estimates generalization error or posterior label error.
- No claim that high prototype influence identifies an incorrect label.
- No claim that the probability model estimates real-world label-error posteriors
  without a separately justified calibration model.
- No claim that generic upper bounds are useful when they are numerically vacuous.
