# Theorem Proofs and Scope

This file is the mathematical contract for the revision implementation.  The
classifier uses fixed training positions, stable distance/index ordering, a
finite class set, and a deterministic class-priority order for ties.

## Definitions

Let `S = ((z_i, y_i))_{i=1}^n`, let `Q = {x_t}_{t=1}^q`, and let
`N_k(x_t)` be the ordered k-neighbor index set.  A replacement `i -> c`
changes only `y_i`, never `z_i`.  For a query, `v_a` is the number of votes
for class `a`, and `pi(a)` is its tie-priority rank (smaller is preferred).

The exact directional influence is

`I_{i->c} = sum_t 1{h_{S^{i->c}}(x_t) != h_S(x_t)}`.

The exact point vulnerability is the indicator that at least one allowed
replacement direction changes the prediction.  These are different objects:
point vulnerability permits a different direction at each query, whereas a
global influence score fixes one prototype and one target class for the whole
query batch.

## Theorem 0: finite LOO blind spot for every k

Fix index-based neighbor ordering and class priority `0 < 1`.  For every
integer `k >= 1`, form a finite weighted graph with query vertex `x`,
representatives `z_1,...,z_k`, and private supports `s_{i,1},...,s_{i,k}`.
The edges are `x--z_i` of weight `1` and `z_i--s_{i,t}` of weight
`epsilon`, where `0 < epsilon < 1/2`; use the shortest-path metric.  The
training sample contains the representatives and supports, but not `x`.
Assign label `0` to `ceil(k/2)` representatives and label `1` to the rest,
and give each support its representative's label.

Deleting a representative leaves its `k` private supports as the nearest
same-label occurrences.  Deleting one support leaves its representative and
the other `k-1` supports in the same cluster as the nearest `k` occurrences.
The graph distances separate each cluster from every other cluster, so every
training-point LOO prediction is correct.  At `x`, the representatives are
the first `k` neighbors because they are at distance `1` and the supports are
at distance `1+epsilon`.  The initial prediction is `0`, by strict majority
for odd `k` and by the fixed priority rule for even `k`.  Relabeling one
label-`0` representative to `1` changes the representative vote to a strict
label-`1` majority.  Thus a fixed-position relabeling changes the prediction at
`x` for every `k >= 1`.

This theorem is the main finite separation result.  It is stronger than a
statement that LOO loss is unchanged: every deleted training occurrence is
correctly recovered.  The two-dimensional ring/support construction below is
an additional Euclidean realization for `k >= 2`, not the source of the
all-`k` claim.

## Proposition 1: exact multiclass one-label test

Assume a query has current winner `a`, with count `v_a`, and a prototype in
the query neighborhood is changed from its current class `b` to `c != b`.
Set `v'_b = v_b - 1`, `v'_c = v_c + 1`, and `v'_d = v_d` otherwise.  The
prediction changes if and only if either

1. `a` is not a maximizer of `v'`, or
2. `a` is a maximizer of `v'` but some other maximizer has strictly smaller
   priority rank than `a`.

Equivalently, the prediction is unchanged if and only if

`v'_a > max_{d != a} v'_d`, or
`v'_a = max_{d != a} v'_d` and `pi(a) < min{pi(d): v'_d = v'_a, d != a}`.

### Proof

Only the two affected vote counts can change.  The deterministic classifier
selects a maximum count and then the minimum priority rank among tied maxima.
Applying this definition to `v'` gives exactly the two cases above.  No
assumption about the number of classes, zero-count competitors, or parity of
`k` is used.  If the allowed replacement set is restricted, the statement is
evaluated only for those `c`; no margin shortcut is valid without that check.

### Corollary: the two-vote screening band

When every prototype in the current winning class may be changed to every
other class, a one-label change cannot flip a query with top-two gap greater
than two.  Gap one is always flippable by changing a winning vote to a
second-place class.  Gap two is flippable exactly when the relevant tied
post-change competitor precedes the old winner under `pi`.  The exact code
still evaluates every allowed direction, because restricted replacement sets
and non-winning-class changes are not covered by this shortcut.

## Proposition 2: single-prototype motion certificate

Under unweighted deterministic voting, move one training prototype to an
arbitrary new position and change its label arbitrarily, while leaving every
other prototype and the query fixed.  If the pre-perturbation top-two vote gap
is strictly greater than two, the prediction cannot change.

### Proof

Only the moved prototype can enter or leave the ordered neighborhood.  Thus the
post-change neighborhood differs by at most one member exchange, and the vote
vector is changed by at most one vote replacement.  The old winner can lose at
most one vote and one competitor can gain at most one vote, so the top-two gap
shrinks by at most two.  A strict gap greater than two preserves the winner.
This is a sufficient certificate, not a converse.

## Proposition 3: binary odd-k pivotal identity

Let `k = 2m+1`, let the two labels be encoded as zero and one, and let a
query contain distinct prototype indices.  Define `P_i(x)=1` when prototype
`i` is in the neighborhood and the other `2m` votes contain exactly `m` votes
for each class.  Then `P_i(x)=1` if and only if flipping label `i` changes the
majority prediction.  Consequently,

`sum_i I_i(S,Q) = (m+1) sum_{x in Q} V_k(S,x) = ((k+1)/2) sum_x V_k(S,x)`.

### Proof

With an odd vote count, the other `2m` votes are tied exactly when the
removed vote is one of the `m+1` votes for the current majority.  Changing
that vote produces `m` votes for the old winner and `m+1` for the other class,
so the prediction changes.  If the other votes are not tied, their majority
remains a majority after one label change.  For every vulnerable query,
exactly its `m+1` current majority prototypes satisfy the condition; a
non-vulnerable query contributes zero.  Summing over queries proves the
identity.

This is the standard pivotal-variable identity for odd majority, specialized
to fixed kNN neighborhoods.  It is a correctness check, not a claim of a new
Boolean-function theorem.

## Proposition 4: point risk and global single-relabel risk

For any binary odd-k batch with `n` training prototypes, let
`R_point = q^{-1} sum_t V_t` and `R_1 = q^{-1} max_{i,c} I_{i->c}`.
When the same opposite-label replacement is admissible for all prototypes,
then

`((k+1)/(2n)) R_point <= R_1 <= R_point`.

### Proof

The upper bound follows because a fixed prototype can be decisive only for a
vulnerable query.  For the lower bound, each vulnerable odd-binary query has
exactly `(k+1)/2` decisive prototypes, and the decisive direction is the
opposite class.  Thus `sum_{i,c} I_{i->c} = q R_point (k+1)/2` and
`max_{i,c} I_{i->c} >= (sum_{i,c} I_{i->c})/n`.

The bounds are generally not equalities.  A query-disjoint construction can
spread decisive incidences over many prototypes, while a shared-neighborhood
construction can concentrate them on one prototype.  The factor `n` is
essential; replacing it by `k` is false whenever the number of queries or
training prototypes is larger than the neighborhood size.  The experiment
records these as structural separation examples rather than treating the two
risks as interchangeable estimators.

For the same binary odd-`k` setting, define directional concentration by
`H = max_i I_i / sum_i I_i`.  The incidence identity gives the exact
normalization `R_1 / R_point = ((k+1)/2) H`.  This is a structural
reparameterization, not an independent empirical correlation.  No scalar
equality is claimed for multiclass replacement sets or even-`k` tie policies.

## Proposition 5: classification-error change

For a fixed replacement direction `i -> c`, with true query labels `y_t^*`,

`DeltaErr_{i->c} = q^{-1} sum_t (1{h_{S^{i->c}}(x_t) != y_t^*} - 1{h_S(x_t) != y_t^*})`.

If the binary prediction necessarily flips on a query under the considered
direction, its contribution is `1 - 2 e_0(x_t)`, where `e_0` is the baseline
error indicator.  Therefore prediction-change risk and error improvement are
not equivalent: a change can repair an error or damage a correct prediction.

## Proposition 6: exact local probability under independent label flips

Let the k fixed neighbors have labels `Y_j` and independent flip indicators
`B_j ~ Bernoulli(p_j)`.  In the binary case, the post-flip number of class-0
votes is Poisson-binomial with success probabilities

`q_j = 1-p_j` when `Y_j=0`, and `q_j=p_j` when `Y_j=1`.

The dynamic program

`D_0(0)=1`, `D_j(s)=D_{j-1}(s)(1-q_j)+D_{j-1}(s-1)q_j`

returns the exact mass of each post-flip vote count.  Summing masses whose
deterministic vote winner differs from the baseline gives the exact query
flip probability.  This remains valid for heterogeneous probabilities,
boundary probabilities 0 or 1, even `k`, and any fixed tie priority.

### Proof

The recurrence is the convolution of independent Bernoulli masses.  Each
post-flip vote vector maps deterministically to a count and then to a winner;
partitioning the sample space by the count and summing the corresponding
masses is exact.

## Proposition 7: first-order label-noise derivative

For any Boolean query-change function `f(B)` with `B_i` independent Bernoulli
coordinates, let `r(p)=E_p[f(B)]`. At the all-zero vector,

`partial r / partial p_i |_{p=0} = f(e_i)-f(0)`.

For a batch risk, this derivative equals `I_i/q` for the relevant binary
single-flip direction.  The implementation exposes the resulting linear
approximation `q^{-1} sum_i p_i I_i`; it is not labelled as an exact finite-p
formula.

### Proof

Condition on all coordinates except `B_i`.  The conditional expectation is
`(1-p_i)f(0)+p_i f(e_i)` at `p_{-i}=0`; differentiating gives the result.

## Proposition 8: batch variance under shared perturbations

Let `R(B)=q^{-1} sum_t Z_t(B)` be the fraction of queries whose predictions
change under one shared flip vector.  Define `d_i` as the number of queries
whose prediction can change when only prototype i's label is changed, allowing
the other labels to be held at any fixed state.  Then changing coordinate i
changes R by at most `c_i=d_i/q`.  For independent `B_i ~ Bernoulli(p_i)`,

`Var(R) <= sum_i p_i(1-p_i)c_i^2`.

### Proof

Let `B^(i)` replace coordinate i by an independent copy.  The Efron--Stein
inequality gives `Var(R) <= 1/2 sum_i E[(R(B)-R(B^(i)))^2]`.  The two values
can differ only on at most `d_i` query indicators and hence the squared
difference is at most `c_i^2` when the two Bernoulli copies differ.  That
event has probability `2p_i(1-p_i)`, yielding the bound.  Query indicators
are therefore not treated as independent.  The exact pairwise dynamic
program in `probability.py` is used when a binary batch permits it.

## Proposition 9: first-order variance expansion at independent-flip probability zero

Let `Z_t(B)` indicate whether query `t` changes under a binary flip vector and
let `R_epsilon=q^{-1} sum_t Z_t(B)`, where the coordinates of `B` are
independent Bernoulli(`epsilon`).  Let `I_i` be the number of query indicators
that change when only prototype `i` is flipped from the observed label.  Then,
for fixed neighborhoods and `epsilon` tending to zero,

`E[R_epsilon] = (epsilon/q) sum_i I_i + O(epsilon^2)`

and

`Var(R_epsilon) = (epsilon/q^2) sum_i I_i^2 + O(epsilon^2)`.

For binary odd `k`, writing `c=(k+1)/2` and using the pivotal incidence
identity gives the audit bounds

`c^2 R_point^2/n <= lim_{epsilon->0} Var(R_epsilon)/epsilon <= c R_point R_1`.

This is a first-order Boolean-influence expansion used as a conditional local
calculation.  It is not claimed as a new noise-stability theorem.

When `R_point > 0`, define the quadratic influence concentration
`H_2 = sum_i (I_i / sum_j I_j)^2`.  The pivotal incidence identity then gives

`lim_{epsilon->0} Var(R_epsilon)/epsilon = (c R_point)^2 H_2`.

Thus two fixed-neighborhood audits with the same query-level vulnerability can
have different leading batch variance when their decisive incidences are
distributed differently across prototypes.  This is a reparameterization of
the first-order Boolean expansion, not an additional noise-stability theorem;
the E3 validation table checks it on multiple binary datasets and fixed
neighborhoods.

### Proof

The probability of exactly one flipped coordinate `i` is
`epsilon(1-epsilon)^(n-1)=epsilon+O(epsilon^2)`, while the probability of two
or more flips is `O(epsilon^2)`.  The all-zero state contributes zero risk.
Summing the risk over the single-flip states yields the expectation formula.
For the second moment, the single-flip state `i` contributes
`epsilon(I_i/q)^2+O(epsilon^2)`, while multi-flip states contribute
`O(epsilon^2)`.  Subtracting the squared expectation, which is itself
`O(epsilon^2)`, gives the variance formula.  For odd binary `k`,
`sum_i I_i=q c R_point`; also `sum_i I_i^2` is bounded below by
`(sum_i I_i)^2/n` and above by `(max_i I_i)(sum_i I_i)`.  Since
`max_i I_i=q R_1`, the displayed bounds follow.

## Proposition 10: heterogeneous local concentration bound

Assume odd `k=2m+1`, conditional independence of neighbor labels given their
features, and `|eta(x)-eta(x')| <= L ||x-x'||` within the neighborhood.  Let
`r_k(x)=max_j ||X_(j)(x)-x||` and
`a=(|eta(x)-1/2|-L r_k(x))_+`.  Then

`Pr(V_k(x)=1 | X_1,...,X_n) <= exp(-2k (a - 1/(2k))_+^2)`.

### Proof

Suppose `eta(x) >= 1/2`; the other case is symmetric.  If `a=0`, the
right-hand side is one and the claim is the trivial probability bound.  If
`a>0`, each neighbor has conditional class-1 probability at least `1/2+a`.
The vulnerability event is contained in `{sum_j Y_j <= (k+1)/2}`.  Its
threshold is at least `k a - 1/2` below the conditional mean.  Hoeffding's
inequality for independent, not necessarily identically distributed Bernoulli
variables gives the stated bound, with the positive part covering the
remaining boundary regime.

The result is deliberately restricted to odd k and conditional independence;
it is not asserted as a universal finite-sample law for arbitrary dependent
training labels.

## Two-dimensional finite construction

For `k >= 2`, take representatives
`r_j=(cos(2 pi j/k), sin(2 pi j/k))` and supports
`s_{j,t}=(1+t epsilon/k)r_j`, `t=1,...,k`.  Every support is assigned the
representative's class.  The nearest cross-cluster representative distance is
`2 sin(pi/k)`.  Every point in a cluster lies within epsilon of its
representative, so every cross-cluster distance is at least
`2 sin(pi/k)-2 epsilon`, while within-cluster distances are at most epsilon.
Thus `epsilon < 2 sin(pi/k)/3` separates clusters.  At the origin, every
representative is closer than every support because the representative radius
is one and support radii exceed one.  For an open ball of radius
`delta < epsilon/(2k)`, the same ordering is preserved by the triangle
inequality.  Deleting any one point leaves k same-label points in its cluster,
so its LOO prediction is correct under the separation condition.  The
representative labels are assigned with a one-vote margin for odd k and a
tie resolved by class priority for even k; flipping the designated
representative changes every query in that central ball.

The code checks the finite inequalities numerically for each selected k.  The
`k=1` case is handled separately in the experiment by two same-label pairs
placed in separated regions, because the ring argument is only for `k >= 2`.

## Results intentionally not promoted to theorems

Monotonicity of empirical PRV in k, a universal high-order truncation error
bound based only on first-order influence, and a universal asymptotic rate for
the 23-dataset panel are not asserted here.  They require additional
distributional assumptions or are tested as empirical questions.
