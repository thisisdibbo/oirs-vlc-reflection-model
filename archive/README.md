# Archived run logs

Verbatim copies of earlier `freeze_run.log` files, taken unchanged from the
repository history. They are kept because the paper quotes measured timings from
them, and timings cannot be reproduced exactly on another machine.
`python verify_paper.py` confirms every quoted timing against these files.

| file | run | host, as logged | what the paper takes from it |
|---|---|---|---|
| `freeze_run_linux_2.log` | original frozen run, Linux | x86_64, NumPy 2.4.4 | closed-form update 64.9 µs; GA 0.263 ms per evaluation and 11.84 s to converge (Sections 5.3 and 5.9, Table 2); the convergence figure, `fig3_convergence.pdf` |
| `freeze_run_i9.log` | after the orientation fix, Intel Core i9-14900HX, Windows 11 | AMD64, NumPy 2.5.3 | closed-form update 39.8 µs; GA 0.113 ms and 5.08 s (the second-machine values in Sections 5.3 and 5.9) |
| `freeze_run_linux_1.log` | first frozen run, Linux | x86_64, NumPy 2.4.4 | GA 0.171 ms per evaluation and 7.69 s to converge: the latency pinned as `TAU_META_S` in `freeze.py` and charged by Days 8 and 9 |

Only the timings in the table are taken from these files. Every other number the
paper cites comes from the current `freeze_run.log`, which `verify_paper.py`
regenerates. The archived logs differ from it wherever the code has changed
since they were written: the two Linux logs predate the orientation fix
described in the main README (their Day 4 results, 2.24–2.58 dB, are that
artefact) and the later extensions of Days 8 and 9, and all three divide Day 8's
recovered fraction by the metaheuristic baseline rather than by the no-IRS link
the paper defines it against.

`freeze_run_linux_1.log` holds a complete frozen pass, which ends at
`ALL_FREEZE_RUNS_COMPLETE`, followed by `sweep_power.py` (added to the freeze
afterwards) and re-runs of Days 8 and 9, appended to the same file.
