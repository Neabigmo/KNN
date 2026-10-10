# Novelty Matrix

The central boundary of the revision is deliberate: training-data influence,
hubness, voting-margin facts, and classical majority/noise results are not
presented as newly invented. The contribution is an exact, direction-resolved
audit of how one specified fixed-position label edit reaches a query batch,
together with a 23-dataset comparison showing that per-query vulnerability and
the maximum reach of one common edit describe different empirical behavior.
Conditional random-label reliability marks the boundary of that single-edit
audit rather than a separate primary contribution.

| Result or object | Classification | What prior work covers | What this revision claims |
|---|---|---|---|
| One-label binary odd-k pivotality | Direct consequence of majority voting | Classical Boolean influence and pivotal-variable theory | A kNN-specific exact implementation and the normalization $R_1=((k+1)/2)R_{point}H_{max}$ used to interpret the 23-dataset audit |
| Multiclass directional replacement rule | New formulation for this fixed-neighborhood model | Multiclass vote-margin certificates and poisoning robustness | Exact rule under explicit allowed-label set and deterministic tie priority; no claim of new poisoning certification |
| Query-level PRV versus maximum single-prototype influence | Core audit representation with an overlapping poisoning objective | Training-data influence covers individual and group effects; hubness work studies frequently reused neighbors; budget-one label poisoning can equal qR1 under aligned conditions | The full directional incidence matrix records a specified replacement label and its exact batch reach; the 23-dataset result characterizes how Rpoint and R1 differ, without claiming that individual-versus-group influence is a new concept |
| Independent label-flip probability for one query | Direct Poisson-binomial derivation | Randomized-smoothing and noisy-label literature provide related robustness models | Exact conditional probability for fixed kNN positions and observed labels, with DP and enumeration checks |
| First-order risk derivative at zero flip probabilities | Direct Boolean influence identity | Classical influence/noise sensitivity | Used as a transparent bridge between deterministic influence and conditional risk, not claimed as a new general theorem |
| Batch-query risk variance | Conditional extension of the single-edit audit | Efron-Stein and bounded-difference tools give generic bounds | Explicit shared-prototype covariance accounting, reported with relative and absolute effect sizes and no deployment-benefit claim |
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
