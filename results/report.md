# Question 9 – VM placement with DE, ABC and GA-PSO

## Problem

```
Physical machines
  PM    CPU (cores)  RAM (GB)  BW (Gbps)  Power (W)
  PM1            16        32         10        400
  PM2            32        64         20        600
  PM3            24        48         10        500
  PM4             8        16          5        250
  PM5            16        64         10        450

Virtual machines
  VM    CPU (cores)  RAM (GB)  BW (Gbps)
  VM1             4         8          1
  VM2             8        12          2
  VM3             8        16          2
  VM4             2         4          1
  VM5            12        16          3
  Total          34        56          9

Encoding
  One gene per VM = the PM it runs on, e.g. [3, 3, 5, 3, 1] puts VM1, VM2, VM4
  on PM3, VM3 on PM5 and VM5 on PM1. Optimisers search x in [0, 5)^5 and
  decode PM = floor(x) + 1.

Fitness (minimise)
  f = servers_used
    + power_of_active_PMs / (total_power + 1)        tie-breaker, always < 1
    + 100 * (n_violations + sum relative overload)   CPU / RAM / BW limits

Global optimum (exhaustive search of 5^5 = 3125 placements)
  [2, 2, 2, 4, 2]  servers = 2, power = 850 W, fitness = 2.3862
  PM2 (600 W): VM1, VM2, VM3, VM5     CPU 32/32 (100%), RAM 52/64 (81%), BW 8/20 (40%)
  PM4 (250 W): VM4                    CPU 2/8 (25%), RAM 4/16 (25%), BW 1/5 (20%)
```

## Experimental setup

30 independent runs per algorithm (seeds 0..29), population 20, budget 2000 fitness evaluations per run.

## Comparison

| Algorithm | Best placement | Best servers | Mean servers | Best power (W) | Mean util (%) | Success rate (%) | Evals to optimum (mean) | Runtime (ms, mean) |
|---|---|---|---|---|---|---|---|---|
| DE | [4, 2, 2, 2, 2] | 2 | 2.0 | 850 | 54.8 | 56.7 | 225.1 | 166.1 |
| ABC | [2, 2, 2, 4, 2] | 2 | 2.0 | 850 | 48.5 | 86.7 | 525.8 | 176.6 |
| GA-PSO | [4, 2, 2, 2, 2] | 2 | 2.0 | 850 | 54.8 | 40.0 | 126.8 | 159.3 |

Full metrics (fitness mean/std, per-resource utilisation, median evaluations, runtime std) are in `summary.csv`. Utilisation is for the best placement, averaged over the active servers. Evaluations/iterations to optimum are averaged over the runs that reached the global optimum (see success rate).

## Best placement found by each algorithm

```
DE: [4, 2, 2, 2, 2]  (2 servers, 850 W)
  PM2 (600 W): VM2, VM3, VM4, VM5     CPU 30/32 (94%), RAM 48/64 (75%), BW 8/20 (40%)
  PM4 (250 W): VM1                    CPU 4/8 (50%), RAM 8/16 (50%), BW 1/5 (20%)

ABC: [2, 2, 2, 4, 2]  (2 servers, 850 W)
  PM2 (600 W): VM1, VM2, VM3, VM5     CPU 32/32 (100%), RAM 52/64 (81%), BW 8/20 (40%)
  PM4 (250 W): VM4                    CPU 2/8 (25%), RAM 4/16 (25%), BW 1/5 (20%)

GA-PSO: [4, 2, 2, 2, 2]  (2 servers, 850 W)
  PM2 (600 W): VM2, VM3, VM4, VM5     CPU 30/32 (94%), RAM 48/64 (75%), BW 8/20 (40%)
  PM4 (250 W): VM1                    CPU 4/8 (50%), RAM 8/16 (50%), BW 1/5 (20%)
```

## Convergence

![convergence](convergence.png)

![convergence per iteration](convergence_iterations.png)
