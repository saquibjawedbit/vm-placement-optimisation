"""Run every optimiser many times, then build the comparison table, plots and report."""

import csv
import io
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from . import ALGORITHMS, data
from .problem import Problem

# categorical slots 1-3 of the validated reference palette (light mode)
COLORS = {"DE": "#2a78d6", "ABC": "#eb6834", "GA-PSO": "#1baf7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TOL = 1e-9


@dataclass
class Summary:
    name: str
    results: list
    optimum: float
    pop_size: int = 20

    def __post_init__(self):
        self.problem = Problem()
        self.evals_list = [self.problem.evaluate_assignment(r.best_assignment) for r in self.results]
        self.best_idx = int(np.argmin([r.best_fitness for r in self.results]))
        self.best = self.results[self.best_idx]
        self.best_eval = self.evals_list[self.best_idx]
        self.success = np.array([r.best_fitness <= self.optimum + TOL for r in self.results])
        # first evaluation at which the run reached the global optimum
        self.hit_eval = np.array([next((k + 1 for k, f in enumerate(r.trace) if f <= self.optimum + TOL), np.nan)
                                  for r in self.results])
        self.hit_iter = np.ceil(self.hit_eval / self.pop_size)

    def row(self):
        servers = np.array([e.servers_used for e in self.evals_list])
        power = np.array([e.power for e in self.evals_list])
        fitness = np.array([r.best_fitness for r in self.results])
        runtime = np.array([r.runtime for r in self.results]) * 1000
        util = self.problem.utilisation(self.best.best_assignment)
        hits = self.hit_eval[~np.isnan(self.hit_eval)]
        return {
            "Algorithm": self.name,
            "Best placement": str(Problem.to_display(self.best.best_assignment)),
            "Best servers": int(servers.min()),
            "Mean servers": round(float(servers.mean()), 2),
            "Best power (W)": int(power.min()),
            "Mean power (W)": round(float(power.mean()), 1),
            "CPU util (%)": round(util["CPU"], 1),
            "RAM util (%)": round(util["RAM"], 1),
            "BW util (%)": round(util["BW"], 1),
            "Mean util (%)": round(util["mean"], 1),
            "Best fitness": round(float(fitness.min()), 4),
            "Mean fitness": round(float(fitness.mean()), 4),
            "Std fitness": round(float(fitness.std()), 4),
            "Success rate (%)": round(100 * float(self.success.mean()), 1),
            "Evals to optimum (mean)": round(float(hits.mean()), 1) if hits.size else "-",
            "Evals to optimum (median)": int(np.median(hits)) if hits.size else "-",
            "Iters to optimum (mean)": round(float(self.hit_iter[~np.isnan(self.hit_iter)].mean()), 1) if hits.size else "-",
            "Runtime (ms, mean)": round(float(runtime.mean()), 1),
            "Runtime (ms, std)": round(float(runtime.std()), 1),
        }


def run_all(runs=30, max_evals=2000, pop_size=20, seed=0, log=print):
    problem = Problem()
    opt_assignment, opt_eval = problem.brute_force()
    log(f"Brute force over {problem.n_pms ** problem.n_vms} placements: optimum "
        f"{Problem.to_display(opt_assignment)} -> {opt_eval.servers_used} servers, "
        f"{opt_eval.power:.0f} W, fitness {opt_eval.fitness:.4f}")

    summaries = {}
    for name, algo in ALGORITHMS.items():
        results = []
        for r in range(runs):
            p = Problem()
            results.append(algo(p, pop_size, max_evals=max_evals, seed=seed + r))
        summaries[name] = Summary(name, results, opt_eval.fitness, pop_size)
        s = summaries[name]
        log(f"{name:7s} {runs} runs: best {Problem.to_display(s.best.best_assignment)} "
            f"fitness {s.best.best_fitness:.4f}, success {100 * s.success.mean():.0f}%")
    return problem, (opt_assignment, opt_eval), summaries


# --- outputs ---------------------------------------------------------------------------

TABLE_COLS = ["Algorithm", "Best placement", "Best servers", "Mean servers", "Best power (W)",
              "Mean util (%)", "Success rate (%)", "Evals to optimum (mean)", "Runtime (ms, mean)"]


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows, cols=None):
    cols = cols or list(rows[0])
    out = io.StringIO()
    out.write("| " + " | ".join(cols) + " |\n")
    out.write("|" + "|".join("---" for _ in cols) + "|\n")
    for r in rows:
        out.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    return out.getvalue()


def text_table(rows, cols):
    widths = [max(len(c), *(len(str(r[c])) for r in rows)) for c in cols]
    line = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(r[c]).ljust(w) for c, w in zip(cols, widths)) for r in rows]
    return "\n".join([line, sep, *body])


def placement_text(problem, assignment):
    load = problem.loads(assignment)
    lines = []
    for pm in np.unique(assignment):
        vms = [problem.vm_names[i] for i in range(problem.n_vms) if assignment[i] == pm]
        cap = problem.capacity[pm]
        use = ", ".join(f"{r} {load[pm, k]:g}/{cap[k]:g} ({100 * load[pm, k] / cap[k]:.0f}%)"
                        for k, r in enumerate(data.RESOURCES))
        lines.append(f"  {problem.pm_names[pm]} ({problem.pm_power[pm]:.0f} W): {', '.join(vms):<22} {use}")
    return "\n".join(lines)


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _traces(s, max_evals):
    return np.array([np.pad(r.trace, (0, max_evals - len(r.trace)), mode="edge")[:max_evals]
                     for r in s.results])


def convergence_figure(summaries, optimum, max_evals):
    """Per-evaluation convergence: mean best-so-far fitness and share of runs at the optimum."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(1, max_evals + 1)
    for name, s in summaries.items():
        traces = _traces(s, max_evals)
        c = COLORS[name]
        a1.plot(x, traces.mean(axis=0), color=c, linewidth=2, label=name)
        reached = (traces <= optimum + TOL).mean(axis=0) * 100
        a2.plot(x, reached, color=c, linewidth=2, label=name)
        a2.annotate(name, (x[-1], reached[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)

    a1.axhline(optimum, color=INK_2, linewidth=1, linestyle="--")
    a1.annotate(f"global optimum {optimum:.4f}", (x[-1], optimum), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK_2)
    a1.set_xscale("log")
    a1.set_ylim(optimum - 0.03, optimum + 1.0)
    a1.set_xlabel("Fitness evaluations (log)")
    a1.set_ylabel("Mean best-so-far fitness")
    a1.set_title(f"Mean best fitness over {len(s.results)} runs", fontsize=11, color=INK, loc="left")
    a1.legend(frameon=False, fontsize=9, labelcolor=INK)
    a2.set_xscale("log")
    a2.set_ylim(0, 105)
    a2.set_xlabel("Fitness evaluations (log)")
    a2.set_ylabel("Runs at global optimum (%)")
    a2.set_title("Share of runs that have found the optimum", fontsize=11, color=INK, loc="left")
    a2.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    return fig


def mean_curve_figure(summaries, problem, max_evals, pop_size, show_iters=60):
    """Per-iteration convergence of the best solution: servers used and power drawn."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    for name, s in summaries.items():
        per_iter = _traces(s, max_evals)[:, pop_size - 1::pop_size][:, :show_iters + 1]
        feasible = per_iter < problem.n_pms + 1                      # no penalty term
        servers = np.where(feasible, np.floor(per_iter), np.nan)
        power = np.where(feasible, (per_iter - servers) * (problem.total_power + 1), np.nan)
        it = np.arange(per_iter.shape[1])
        a1.plot(it, np.nanmean(servers, axis=0), color=COLORS[name], linewidth=2, label=name)
        a2.plot(it, np.nanmean(power, axis=0), color=COLORS[name], linewidth=2, label=name)
    a1.set_ylabel("Servers used (mean of best)")
    a1.set_title("Servers used by best solution", fontsize=11, color=INK, loc="left")
    a2.set_ylabel("Power of active servers, W (mean of best)")
    a2.set_title("Power drawn by best solution", fontsize=11, color=INK, loc="left")
    for ax in (a1, a2):
        ax.set_xlabel(f"Iteration ({pop_size} evaluations each)")
        ax.set_xlim(0, show_iters)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
        _style(ax)
    fig.tight_layout()
    return fig


def _text_page(pdf, title, body, size=9):
    fig = plt.figure(figsize=(8.27, 11.69))   # A4 portrait
    fig.text(0.07, 0.95, title, fontsize=15, weight="bold", color=INK, va="top")
    fig.text(0.07, 0.91, body, fontsize=size, family="monospace", color=INK, va="top")
    pdf.savefig(fig)
    plt.close(fig)


def problem_text(problem, opt):
    lines = ["Physical machines", f"  {'PM':<5}{'CPU (cores)':>12}{'RAM (GB)':>10}{'BW (Gbps)':>11}{'Power (W)':>11}"]
    for n, (c, r, b, w) in data.PMS.items():
        lines.append(f"  {n:<5}{c:>12}{r:>10}{b:>11}{w:>11}")
    lines += ["", "Virtual machines", f"  {'VM':<5}{'CPU (cores)':>12}{'RAM (GB)':>10}{'BW (Gbps)':>11}"]
    for n, (c, r, b) in data.VMS.items():
        lines.append(f"  {n:<5}{c:>12}{r:>10}{b:>11}")
    tot = problem.demand.sum(axis=0)
    lines += [f"  {'Total':<5}{tot[0]:>12g}{tot[1]:>10g}{tot[2]:>11g}", ""]
    lines += [
        "Encoding",
        "  One gene per VM = the PM it runs on, e.g. [3, 3, 5, 3, 1] puts VM1, VM2, VM4",
        "  on PM3, VM3 on PM5 and VM5 on PM1. Optimisers search x in [0, 5)^5 and",
        "  decode PM = floor(x) + 1.",
        "",
        "Fitness (minimise)",
        "  f = servers_used",
        "    + power_of_active_PMs / (total_power + 1)        tie-breaker, always < 1",
        "    + 100 * (n_violations + sum relative overload)   CPU / RAM / BW limits",
        "",
        "Global optimum (exhaustive search of 5^5 = 3125 placements)",
        f"  {Problem.to_display(opt[0])}  servers = {opt[1].servers_used}, power = {opt[1].power:.0f} W,"
        f" fitness = {opt[1].fitness:.4f}",
        placement_text(problem, opt[0]),
    ]
    return "\n".join(lines)


def write_report(outdir, problem, opt, summaries, rows, cfg):
    conv = convergence_figure(summaries, opt[1].fitness, cfg["max_evals"])
    conv.savefig(f"{outdir}/convergence.png", dpi=160)
    iters = mean_curve_figure(summaries, problem, cfg["max_evals"], cfg["pop_size"])
    iters.savefig(f"{outdir}/convergence_iterations.png", dpi=160)

    write_csv(rows, f"{outdir}/summary.csv")
    placements = "\n\n".join(f"{n}: {Problem.to_display(s.best.best_assignment)}  "
                             f"({s.best_eval.servers_used} servers, {s.best_eval.power:.0f} W)\n"
                             + placement_text(problem, s.best.best_assignment)
                             for n, s in summaries.items())
    setup = (f"{cfg['runs']} independent runs per algorithm (seeds {cfg['seed']}..{cfg['seed'] + cfg['runs'] - 1}),"
             f" population {cfg['pop_size']}, budget {cfg['max_evals']} fitness evaluations per run.")

    with open(f"{outdir}/report.md", "w") as f:
        f.write("# Question 9 – VM placement with DE, ABC and GA-PSO\n\n")
        f.write("## Problem\n\n```\n" + problem_text(problem, opt) + "\n```\n\n")
        f.write("## Experimental setup\n\n" + setup + "\n\n")
        f.write("## Comparison\n\n" + markdown_table(rows, TABLE_COLS) + "\n")
        f.write("Full metrics (fitness mean/std, per-resource utilisation, median evaluations, "
                "runtime std) are in `summary.csv`. Utilisation is for the best placement, averaged over "
                "the active servers. Evaluations/iterations to optimum are averaged over the runs that "
                "reached the global optimum (see success rate).\n\n")
        f.write("## Best placement found by each algorithm\n\n```\n" + placements + "\n```\n\n")
        f.write("## Convergence\n\n![convergence](convergence.png)\n\n"
                "![convergence per iteration](convergence_iterations.png)\n")

    with PdfPages(f"{outdir}/report.pdf") as pdf:
        _text_page(pdf, "Q9 – VM placement: problem and model", problem_text(problem, opt))
        tables = [
            ("Servers and power", ["Algorithm", "Best servers", "Mean servers", "Best power (W)",
                                   "Mean power (W)", "Best fitness", "Mean fitness", "Std fitness"]),
            ("Server utilisation", ["Algorithm", "CPU util (%)", "RAM util (%)", "BW util (%)", "Mean util (%)"]),
            ("Convergence speed and run time", ["Algorithm", "Success rate (%)", "Evals to optimum (mean)",
                                                "Iters to optimum (mean)", "Runtime (ms, mean)"]),
        ]
        note = ("Utilisation: best placement, mean over active servers. Evals/iters to optimum:\n"
                "mean over the runs that reached the global optimum (see success rate).")
        body = (setup + "\n" + note + "\n\n"
                + "\n\n".join(t + "\n" + text_table(rows, c) for t, c in tables)
                + "\n\nBest placement found by each algorithm\n\n" + placements)
        _text_page(pdf, "Q9 – comparison of DE, ABC and GA-PSO", body, size=7.5)
        for fig in (conv, iters):
            pdf.savefig(fig)
            plt.close(fig)
