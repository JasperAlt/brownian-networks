This repository contains code for simulating brownian motion, extracting spatiotemporal networks from the results, and analyzing the networks.

The file `sim.py` contains code for simulating motion and extracting networks using either an R-Tree (the `scan` and `scan_multi` functions) or cell lists (the `scan_cells` and `scan_cells_multi` functions). The cell list is faster for toroids and spaces with dimension less than 3. The worst case running time for both algorithms is n^2.

The file `analysis.py` contains functions for analysis, including for several attributes explored in Y. Peres et al. The file `figs.py` contains functions for visualization and diagnostics.

The file `run.py` is for testing.
