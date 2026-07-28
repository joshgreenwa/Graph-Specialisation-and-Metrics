# Vendored reference data

The three CSVs are the plotted series behind Figure 3(a) of Bamberger et al. (ICML 2025), taken
verbatim from the authors' repository <https://github.com/BenGutteridge/range-measure> at
`data/plotting/grid_task_range_{dirac,rectangle,power}.csv`. The column compared against is
`Val/TaskRangeSPDNorm`.

`grid_task_range_power.csv` is the **self-loop** variant: its own `Name` column reads
`distance_fn: adjacency_self_loop_power_sym_<k>`, which is what identifies which `k-Power`
definition Figure 3 actually plots.

`run_replication.py` hard-codes these values in `PUBLISHED_GRID`; `verify_reference.py` checks the
hard-coded copy against these files so the two cannot drift apart.
