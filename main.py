"""Question 9 demo: place 5 VMs on as few of 5 PMs as possible with DE, ABC and GA-PSO.

    python main.py                 # 30 runs each, writes results/ (table, graphs, report.pdf)
    python main.py --runs 10 --evals 1000
"""

import argparse
import os

from vm_placement import ALGORITHMS, Problem
from vm_placement.experiment import TABLE_COLS, placement_text, run_all, text_table, write_report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=30, help="independent runs per algorithm")
    ap.add_argument("--evals", type=int, default=2000, help="fitness evaluations per run")
    ap.add_argument("--pop", type=int, default=20, help="population / colony / swarm size")
    ap.add_argument("--seed", type=int, default=0, help="seed of the first run")
    ap.add_argument("--out", default="results", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("=" * 78)
    log("Question 9 - VM placement with DE, ABC and hybrid GA-PSO")
    log("=" * 78)
    problem, opt, summaries = run_all(args.runs, args.evals, args.pop, args.seed, log=log)

    log("\nBest placement per algorithm")
    for name, s in summaries.items():
        e = s.best_eval
        log(f"\n{name}: {Problem.to_display(s.best.best_assignment)} -> {e.servers_used} servers, "
            f"{e.power:.0f} W, feasible={e.feasible}")
        log(placement_text(problem, s.best.best_assignment))

    rows = [s.row() for s in summaries.values()]
    log("\nComparison table")
    log(text_table(rows, TABLE_COLS))

    cfg = {"runs": args.runs, "max_evals": args.evals, "pop_size": args.pop, "seed": args.seed}
    write_report(args.out, problem, opt, summaries, rows, cfg)
    log(f"\nWrote {args.out}/summary.csv, report.md, report.pdf, convergence.png, convergence_iterations.png")
    with open(os.path.join(args.out, "output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
