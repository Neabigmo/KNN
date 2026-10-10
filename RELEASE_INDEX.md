# Release index

## `paa-revision-2026-10-10-v12`

This tag is the synchronized public implementation for the October 10, 2026
manuscript revision. It includes the corrected E6 geometry audit, the Fig. 5
median/Q25--Q75 panel construction, controlled shared/distributed variance
validation, class-by-$k$ vulnerability summaries, geometry-operation panels,
and the common-direction runtime ratio panels with final-size layout checks.
The figure builder keeps plotted rows and validation supplements distinct.
The tag contains source code and reproducibility instructions; benchmark
arrays and private frozen audit caches are acquired or supplied separately.

## `paa-revision-2026-10-10-v14-reframed`

This tag adds the E3 dependence-gap audit and the exact finite-state
counterexample enumeration used by the reframed manuscript. It also records
the corrected runtime interpretation: the factorized implementation does not
show a consistent advantage across the tested configurations.

## `paa-revision-2026-10-10-v15-all-k-reframed`

This tag synchronizes the public theory notes with the submission package. It
records the finite graph-metric construction for every `k >= 1`, in which all
training-point LOO predictions are correct while a fixed-position prototype
relabeling changes a separate query. The E3 analysis programs and frozen
implementation from v14 are unchanged.
