"""Figures for the range-measure comparison.

Categorical colour is never asked to carry more than three identities at once: the task variants
are faceted into small multiples, and within a panel the identities are the two (or three)
*measures*.  Every series also carries a distinct line style and marker, so identity never rests
on hue alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RESULTS = HERE / "results"

# Validated categorical slots 1-3 (blue / orange / aqua) plus recessive ink.
JACOBIAN = "#2a78d6"
CARRIAGE = "#eb6834"
REFERENCE = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#8a8a86"
GRID = "#dedddb"

TASK_LABELS = {
    "dirac": "$k$-Dirac",
    "rectangle": "$k$-Rectangle",
    "power_loops": "$k$-Power",
    "power": "$k$-Power (no self-loops)",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#3a3a38",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.6,
            "ytick.major.size": 2.6,
            "lines.linewidth": 1.4,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.dpi": 350,
        }
    )


def panel_letter(ax, letter: str, dx: float = -0.22, dy: float = 1.12) -> None:
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _band(ax, x, entries, colour):
    low = [e["low"] for e in entries]
    high = [e["high"] for e in entries]
    ax.fill_between(x, low, high, color=colour, alpha=0.16, linewidth=0)


# --------------------------------------------------------------------------------------


def figure_replication(name: str = "replication.json", stem: str = "fig1_replication") -> Path:
    data = load(name)
    rows = data["results"]
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    influence = data["influence_panels"]
    influence_tasks = [t for t in ("dirac", "rectangle", "power_loops") if t in influence]

    fig = plt.figure(figsize=(7.2, 4.1))
    grid = fig.add_gridspec(2, 4, hspace=0.62, wspace=0.42)

    for column, task in enumerate(tasks):
        ax = fig.add_subplot(grid[0, column])
        subset = [r for r in rows if r["task"] == task]
        k = [r["k"] for r in subset]
        jac = [r["jacobian"] for r in subset]
        car = [r["carriage_normalised"] for r in subset]
        published = [r.get("published") for r in subset]
        if all(value is not None for value in published):
            ax.plot(
                k,
                published,
                color=MUTED,
                linewidth=3.2,
                alpha=0.55,
                solid_capstyle="round",
                label="Published (Fig. 3a)",
                zorder=2,
            )
        _band(ax, k, jac, JACOBIAN)
        _band(ax, k, car, CARRIAGE)
        ax.plot(
            k,
            [e["estimate"] for e in jac],
            color=JACOBIAN,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Jacobian range",
            zorder=3,
        )
        ax.plot(
            k,
            [e["estimate"] for e in car],
            color=CARRIAGE,
            linestyle=(0, (3.5, 2.2)),
            marker="s",
            markersize=3.0,
            label="Carriage (event-norm.)",
            zorder=4,
        )
        ax.set_title(TASK_LABELS[task], pad=4)
        ax.set_xlabel("$k$")
        ax.set_xticks([2, 4, 6, 8])
        ax.grid(axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel(r"Range  $\hat{\rho}^{\mathrm{spd}}_{\mathcal{G}}$")
            panel_letter(ax, "a", dx=-0.42)
            ax.legend(
                frameon=False, loc="lower right", handlelength=1.5, borderpad=0.1,
                fontsize=6.3, labelspacing=0.25,
            )

    max_distance = 5
    for column, task in enumerate(influence_tasks):
        ax = fig.add_subplot(grid[1, column])
        entry = influence[task]
        distance = np.asarray(entry["distance"])
        bins = np.arange(0, max_distance + 1)
        # Mean per node within each shell: on a 2-D grid the shells grow with distance, and the
        # paper's schematic describes the per-node shape (spike / flat / decaying), not shell mass.
        jac_mass = np.asarray(
            [float(np.asarray(entry["jacobian"])[distance == d].mean()) for d in bins]
        )
        car_mass = np.asarray(
            [float(np.asarray(entry["carriage"])[distance == d].mean()) for d in bins]
        )
        ax.bar(
            bins,
            jac_mass,
            width=0.62,
            color=JACOBIAN,
            alpha=0.85,
            linewidth=0,
            label="Jacobian $I_u(v)$",
        )
        ax.plot(
            bins,
            car_mass,
            linestyle="none",
            marker="o",
            markersize=4.4,
            markerfacecolor="none",
            markeredgecolor=CARRIAGE,
            markeredgewidth=1.2,
            label="Carriage (event-norm.)",
        )
        ax.set_title(TASK_LABELS[task], pad=4)
        ax.set_xlabel("Hop distance from $u$")
        ax.set_xticks(bins)
        ax.set_ylim(0, max(1.05 * float(jac_mass.max()), 1e-3))
        ax.grid(axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        if column == 0:
            ax.set_ylabel("Influence per node")
            panel_letter(ax, "b", dx=-0.42)
            ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderpad=0.1)

    ax = fig.add_subplot(grid[1, 3])
    xs = np.asarray([r["jacobian"]["estimate"] for r in rows])
    ys = np.asarray([r["carriage_normalised"]["estimate"] for r in rows])
    limit = float(max(xs.max(), ys.max())) * 1.08
    ax.plot([0, limit], [0, limit], color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)
    ax.plot(
        xs,
        ys,
        linestyle="none",
        marker="o",
        markersize=4.0,
        markerfacecolor=CARRIAGE,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )
    worst = float(np.abs(xs - ys).max())
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("Jacobian range")
    ax.set_ylabel("Carriage range")
    ax.set_title(f"{len(rows)} task$\\times k$ settings", pad=4)
    note = f"max $|\\Delta|$ = {worst:.1e}"
    against = [
        abs(r["published"] - r["carriage_normalised"]["estimate"])
        for r in rows
        if r.get("published") is not None
    ]
    if against:
        note += f"\nvs published: {max(against):.1e}"
    ax.text(0.05, 0.92, note, transform=ax.transAxes, fontsize=6.6, color=INK, va="top")
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.36)

    out = FIGURES / f"{stem}.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------


def figure_exactness() -> Path:
    data = load("replication.json")
    rows = data["results"]
    tasks = list(dict.fromkeys(r["task"] for r in rows))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.15))
    fig.subplots_adjust(wspace=0.60)

    ax = axes[0]
    markers = ["o", "s", "^", "D"]
    for task, marker in zip(tasks, markers):
        subset = [r for r in rows if r["task"] == task]
        ax.plot(
            [r["k"] for r in subset],
            [max(r["influence_max_abs_error"], 1e-18) for r in subset],
            marker=marker,
            markersize=3.4,
            linewidth=1.0,
            color=CARRIAGE,
            alpha=0.75,
            label=TASK_LABELS[task],
        )
    ax.axhline(np.finfo(np.float64).eps, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.annotate(
        "float64 $\\epsilon$",
        xy=(8, np.finfo(np.float64).eps),
        xytext=(-2, 4),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
        color=MUTED,
    )
    ax.set_yscale("log")
    worst = max(r["influence_max_abs_error"] for r in rows)
    # Never clip: the panel exists to catch a large residual, so let one push the axis.
    ax.set_ylim(1e-18, max(1e-10, 10 ** np.ceil(np.log10(max(worst, 1e-18)) + 1)))
    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\max_{i,s}\,|\tilde{F}[i,s]-|L_{is}||$")
    ax.set_title("Recovery of the influence matrix", pad=4)
    ax.legend(
        frameon=False, ncol=2, handlelength=1.1, borderpad=0.1, loc="lower left",
        fontsize=5.8, columnspacing=0.8, labelspacing=0.25,
    )
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.40)

    ax = axes[1]
    for task, marker in zip(tasks, markers):
        subset = [r for r in rows if r["task"] == task]
        jac = np.asarray([r["jacobian"]["estimate"] for r in subset])
        raw = np.asarray([r["carriage_raw"]["estimate"] for r in subset])
        ax.plot(
            [r["k"] for r in subset],
            100.0 * (raw - jac) / jac,
            marker=marker,
            markersize=3.4,
            linewidth=1.0,
            color=JACOBIAN,
            alpha=0.75,
            label=TASK_LABELS[task],
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.legend(
        frameon=False, ncol=2, handlelength=1.1, borderpad=0.1, loc="lower left",
        fontsize=5.8, columnspacing=0.8, labelspacing=0.25,
    )
    ax.set_xlabel("$k$")
    ax.set_ylabel("Deviation from Jacobian (%)")
    ax.set_title("Un-normalised $F_{\\mathrm{sens}}$: donor scaling", pad=4)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.40)

    ax = axes[2]
    try:
        sweep = load("donor_sweep.json")
    except FileNotFoundError:
        ax.axis("off")
    else:
        donors = [row["donors"] for row in sweep["results"]]
        jac = sweep["jacobian"]
        norm = np.asarray([row["carriage_normalised"]["estimate"] for row in sweep["results"]])
        raw = np.asarray([row["carriage_raw"]["estimate"] for row in sweep["results"]])
        ax.axhline(jac, color=JACOBIAN, linewidth=1.2, label="Jacobian range")
        ax.plot(
            donors,
            norm,
            color=CARRIAGE,
            marker="s",
            markersize=3.4,
            linestyle=(0, (3.5, 2.2)),
            label="Event-normalised",
        )
        ax.plot(
            donors,
            raw,
            color=REFERENCE,
            marker="^",
            markersize=3.6,
            linestyle=(0, (1.2, 1.6)),
            label="Un-normalised",
        )
        het = sweep.get("heteroscedastic")
        if het:
            ax.plot(
                [row["donors"] for row in het],
                [row["carriage_raw"]["estimate"] for row in het],
                color=REFERENCE,
                marker="v",
                markersize=3.6,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linestyle=(0, (0.8, 1.4)),
                alpha=0.95,
                label="Un-normalised,\nheteroscedastic",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(donors)
        ax.set_xticklabels([str(d) for d in donors])
        ax.set_xlabel("Donor events per source $K$")
        ax.set_ylabel("Range")
        ax.set_title(f"{TASK_LABELS[sweep['task']]}, $k={sweep['k']}$", pad=4)
        ax.legend(
            frameon=False, handlelength=1.3, borderpad=0.1, fontsize=5.6,
            labelspacing=0.2, handletextpad=0.4,
        )
        ax.grid(axis="y", alpha=0.9)
        ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.40)

    out = FIGURES / "fig2_exactness.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------


def figure_learned() -> Path:
    data = load("learned.json")
    rows = data["results"]
    tasks = list(dict.fromkeys(r["task"] for r in rows))

    fig, axes = plt.subplots(1, len(tasks) + 1, figsize=(7.2, 2.15))
    fig.subplots_adjust(wspace=0.62)

    for index, task in enumerate(tasks):
        ax = axes[index]
        subset = [r for r in rows if r["task"] == task]
        k = [r["k"] for r in subset]
        ax.plot(
            k,
            [r["operator_range"] for r in subset],
            color=MUTED,
            linewidth=1.0,
            linestyle=(0, (2, 2)),
            marker="",
            label="Operator (target)",
            zorder=1,
        )
        jac = [r["jacobian"] for r in subset]
        car = [r["carriage_normalised"] for r in subset]
        _band(ax, k, jac, JACOBIAN)
        _band(ax, k, car, CARRIAGE)
        ax.plot(
            k,
            [e["estimate"] for e in jac],
            color=JACOBIAN,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Jacobian range",
            zorder=3,
        )
        ax.plot(
            k,
            [e["estimate"] for e in car],
            color=CARRIAGE,
            linestyle=(0, (3.5, 2.2)),
            marker="s",
            markersize=3.0,
            label="Carriage (event-norm.)",
            zorder=4,
        )
        for row in subset:
            if not np.isfinite(row["carriage_normalised"]["estimate"]):
                ax.annotate(
                    "not learnable",
                    xy=(row["k"], row["operator_range"]),
                    xytext=(-2, -26),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=5.8,
                    color=MUTED,
                )
                ax.plot(
                    [row["k"]], [row["operator_range"]], marker="x", markersize=5,
                    color=MUTED, markeredgewidth=1.2, linestyle="none",
                )
        ax.set_title(TASK_LABELS[task], pad=4)
        ax.set_xlabel("$k$")
        ax.set_xticks(k)
        ax.grid(axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        if index == 0:
            ax.set_ylabel("Range of the trained model")
            ax.legend(
                frameon=False, loc="upper left", handlelength=1.0, borderpad=0.1,
                fontsize=5.6, labelspacing=0.2, handletextpad=0.4,
            )
            panel_letter(ax, "a", dx=-0.42)

    ax = axes[-1]
    residual_j, residual_c, r2, dropped = [], [], [], 0
    for row in rows:
        if not np.isfinite(row["carriage_normalised"]["estimate"]):
            dropped += 1
            continue
        residual_j.append(row["jacobian"]["estimate"] - row["operator_range"])
        residual_c.append(row["carriage_normalised"]["estimate"] - row["operator_range"])
        r2.append(row["fit"]["test_r2"])
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.plot(
        r2,
        residual_j,
        linestyle="none",
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=JACOBIAN,
        markeredgewidth=1.2,
        label="Jacobian",
    )
    ax.plot(
        r2,
        residual_c,
        linestyle="none",
        marker="s",
        markersize=3.4,
        color=CARRIAGE,
        label="Carriage (event-norm.)",
    )
    ax.set_xlabel("Model fit $R^2$")
    ax.set_ylabel("Measured $-$ operator range")
    ax.set_title("Error against the target", pad=4)
    if dropped:
        ax.text(
            0.06, 0.52, f"{dropped} setting\nnon-estimable", transform=ax.transAxes,
            fontsize=6.0, color=MUTED, va="center",
        )
    ax.legend(frameon=False, handlelength=1.2, borderpad=0.1)
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.42)

    out = FIGURES / "fig3_learned.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


# --------------------------------------------------------------------------------------


def figure_divergence() -> Path:
    data = load("divergence.json")
    sweeps = data["sweeps"]

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.4))
    fig.subplots_adjust(wspace=0.38, hspace=0.62)
    axes = axes.reshape(-1)

    ax = axes[0]
    rows = sweeps["dose"]
    alpha = [r["alpha"] for r in rows]
    jac = [r["jacobian"]["estimate"] for r in rows]
    car = [r["carriage"] for r in rows]
    ax.plot(alpha, jac, color=JACOBIAN, linewidth=1.4, label="Jacobian range")
    _band(ax, alpha, car, CARRIAGE)
    ax.plot(
        alpha,
        [e["estimate"] for e in car],
        color=CARRIAGE,
        linestyle=(0, (3.5, 2.2)),
        marker="s",
        markersize=3.4,
        label="Carriage (event-norm.)",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Donor dose fraction $\alpha$")
    ax.set_ylabel("Range")
    ax.set_title("Tangent is the $\\alpha\\to 0$ limit", pad=4)
    ax.legend(frameon=False, handlelength=1.7, borderpad=0.1, loc="upper left")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.34)

    for index, (key, xlabel, title, invert) in enumerate(
        (
            ("step", r"Step width $\tau$", "Conditionally active far term", True),
            ("oscillate", r"Frequency $\omega$", "High-frequency far term", False),
        )
    ):
        ax = axes[index + 1]
        rows = sweeps[key]
        x = [r["parameter"] for r in rows]
        jac = [r["jacobian"] for r in rows]
        car = [r["carriage"] for r in rows]
        _band(ax, x, jac, JACOBIAN)
        _band(ax, x, car, CARRIAGE)
        ax.plot(
            x,
            [e["estimate"] for e in jac],
            color=JACOBIAN,
            marker="o",
            markersize=4.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Jacobian range",
        )
        ax.plot(
            x,
            [e["estimate"] for e in car],
            color=CARRIAGE,
            linestyle=(0, (3.5, 2.2)),
            marker="s",
            markersize=3.4,
            label="Carriage (event-norm.)",
        )
        ax.plot(
            x,
            [r["carriage_raw"]["estimate"] for r in rows],
            color=CARRIAGE,
            linestyle=(0, (1.0, 1.4)),
            marker="s",
            markersize=3.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            alpha=0.95,
            label="Carriage (raw $F_{\\mathrm{sens}}$)",
        )
        ax.plot(
            x,
            [r["shell_reference"] for r in rows],
            color=REFERENCE,
            linestyle=(0, (1.2, 1.6)),
            marker="^",
            markersize=3.6,
            label="Gradient-free probe",
        )
        ax.set_xscale("log")
        if invert:
            ax.invert_xaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=4)
        ax.grid(axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        ax.set_ylabel("Range")
        if index == 0:
            ax.legend(
                frameon=False, handlelength=1.5, borderpad=0.1, loc="lower left",
                fontsize=5.6, labelspacing=0.2, handletextpad=0.4,
            )
        panel_letter(ax, "bc"[index], dx=-0.34)

    ax = axes[3]
    rows = sweeps["channels"]
    channels = [r["channels"] for r in rows]
    jac = [r["jacobian"] for r in rows]
    car = [r["carriage"] for r in rows]
    _band(ax, channels, jac, JACOBIAN)
    _band(ax, channels, car, CARRIAGE)
    ax.plot(
        channels,
        [e["estimate"] for e in jac],
        color=JACOBIAN,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label="Jacobian range",
    )
    ax.plot(
        channels,
        [e["estimate"] for e in car],
        color=CARRIAGE,
        linestyle=(0, (3.5, 2.2)),
        marker="s",
        markersize=3.4,
        label="Carriage (event-norm.)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(channels)
    ax.set_xticklabels([str(c) for c in channels])
    ax.set_xlabel("Feature channels $d$")
    ax.set_ylabel("Range")
    ax.set_title("Diffuse far block: $L_1$ vs $L_2$", pad=4)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "d", dx=-0.34)

    out = FIGURES / "fig4_divergence.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_graphlevel() -> Path:
    data = load("graphlevel.json")
    rows = data["results"]
    profiles = data["profiles"]

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.15))
    fig.subplots_adjust(wspace=0.44)

    ax = axes[0]
    k = [r["k"] for r in rows]
    hess = [r["hessian"] for r in rows]
    car = [r["carriage"] for r in rows]
    _band(ax, k, hess, JACOBIAN)
    _band(ax, k, car, CARRIAGE)
    ax.plot(
        k,
        [e["estimate"] for e in hess],
        color=JACOBIAN,
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=r"Hessian range $\hat{\eta}_{\mathcal{G}}$",
    )
    ax.plot(
        k,
        [e["estimate"] for e in car],
        color=CARRIAGE,
        linestyle=(0, (3.5, 2.2)),
        marker="s",
        markersize=3.4,
        label="Carriage (event-norm.)",
    )
    ax.set_xlabel("$k$")
    ax.set_ylabel("Range")
    ax.set_xticks(k)
    ax.set_title("Graph-level pairwise task", pad=4)
    ax.legend(frameon=False, handlelength=1.7, borderpad=0.1, loc="upper left")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.32)

    ax = axes[1]
    steps = np.asarray(profiles["distance"])
    keep = steps <= profiles["k"] + 2
    ax.bar(
        steps[keep],
        np.asarray(profiles["hessian"])[keep],
        width=0.62,
        color=JACOBIAN,
        alpha=0.85,
        linewidth=0,
        label="Hessian $|\\partial^2 y/\\partial x_u \\partial x_v|$",
    )
    ax.plot(
        steps[keep],
        np.asarray(profiles["carriage"])[keep],
        linestyle="none",
        marker="o",
        markersize=4.4,
        markerfacecolor="none",
        markeredgecolor=CARRIAGE,
        markeredgewidth=1.2,
        label="Carriage (event-norm.)",
    )
    ax.set_xlabel("Hop distance")
    ax.set_ylabel("Influence per pair")
    ax.set_xticks(steps[keep])
    ax.set_title(f"Profiles at $k={profiles['k']}$", pad=4)
    ax.legend(frameon=False, handlelength=1.2, borderpad=0.1, fontsize=6.4)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.32)

    out = FIGURES / "fig5_graphlevel.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_realmodel() -> Path:
    data = load("realmodel.json")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    fig.subplots_adjust(wspace=0.50)

    ax = axes[0]
    rows = data["dose_ladder"]
    alpha = [r["alpha"] for r in rows]
    car = [r["carriage"] for r in rows]
    jac = data["jacobian"]
    ax.axhspan(jac["low"], jac["high"], color=JACOBIAN, alpha=0.16, linewidth=0)
    ax.axhline(jac["estimate"], color=JACOBIAN, linewidth=1.4, label="Jacobian range")
    _band(ax, alpha, car, CARRIAGE)
    ax.plot(
        alpha,
        [e["estimate"] for e in car],
        color=CARRIAGE,
        linestyle=(0, (3.5, 2.2)),
        marker="s",
        markersize=3.6,
        label="Carriage (event-norm.)",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"Donor dose fraction $\alpha$")
    ax.set_ylabel("Range (hops)")
    ax.set_title("Trained model, real task", pad=4)
    ax.legend(frameon=False, handlelength=1.5, borderpad=0.1, fontsize=6.3, loc="lower left")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.38)

    ax = axes[1]
    att = {int(k): v for k, v in data["attenuation_by_distance"].items()}
    steps = [k for k in sorted(att) if k <= 8]
    ax.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.plot(
        steps,
        [att[k] for k in steps],
        color=CARRIAGE,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )
    ax.set_xlabel("Hop distance from the swapped node")
    ax.set_ylabel("Finite / tangent influence")
    ax.set_title("Near amplified, far attenuated", pad=4)
    ax.set_xticks(steps)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.38)

    ax = axes[2]
    ref = np.asarray(data["task_reference_per_graph"])
    ax.plot(
        ref,
        data["jacobian_per_graph"],
        linestyle="none",
        marker="o",
        markersize=4.2,
        markerfacecolor="white",
        markeredgecolor=JACOBIAN,
        markeredgewidth=1.2,
        label=f"Jacobian  $r_s$={data['agreement']['spearman_jacobian_vs_reference']:+.2f}",
    )
    ax.plot(
        ref,
        data["carriage_per_graph"],
        linestyle="none",
        marker="s",
        markersize=3.6,
        color=CARRIAGE,
        label=f"Carriage  $r_s$={data['agreement']['spearman_carriage_vs_reference']:+.2f}",
    )
    ax.set_xlabel("Task-required range (hops)")
    ax.set_ylabel("Measured range (hops)")
    ax.set_title("Against the task's own structure", pad=4)
    ax.legend(frameon=False, handlelength=1.0, borderpad=0.1, fontsize=6.0, loc="upper left")
    ax.grid(alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.38)

    out = FIGURES / "fig6_realmodel.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_beneficial() -> Path:
    data = load("beneficial.json")
    splits = ["id", "ood"]
    labels = {
        "dose_null": "Donor dose (model-free)",
        "functional_raw": "Functional $F_{\\mathrm{sens}}$ (raw)",
        "beneficial": "Beneficial carriage",
        "functional": "Functional (event-norm.)",
        "jacobian": "Jacobian influence",
    }
    order = ["dose_null", "functional_raw", "beneficial", "functional", "jacobian"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    fig.subplots_adjust(wspace=0.52)

    ax = axes[0]
    width = 0.36
    positions = np.arange(len(order))
    for offset, split, colour, hatch in (
        (-width / 2, "id", CARRIAGE, None),
        (width / 2, "ood", JACOBIAN, "///"),
    ):
        ax.bar(
            positions + offset,
            [data[split]["auroc"][key] for key in order],
            width=width,
            color=colour,
            alpha=0.85,
            linewidth=0,
            hatch=hatch,
            label="in-distribution" if split == "id" else "out-of-distribution",
        )
    ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(len(order) - 0.55, 0.52, "chance", fontsize=6.0, color=MUTED, ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[key] for key in order], rotation=32, ha="right", fontsize=6.0)
    ax.set_ylabel("AUROC: marks vs ordinary")
    ax.set_ylim(0, 1.05)
    ax.set_title("Confounded: a model-free null wins", pad=4)
    ax.legend(frameon=False, fontsize=6.0, handlelength=1.0, borderpad=0.1, loc="upper right")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.40)

    ax = axes[1]
    for split, colour, marker, style_ in (("id", CARRIAGE, "s", (0, (3.5, 2.2))), ("ood", JACOBIAN, "o", "-")):
        bins = {int(k): v for k, v in data[split]["S_B"].items()}
        steps = sorted(bins)
        ax.plot(
            range(len(steps)),
            [bins[k] for k in steps],
            color=colour,
            marker=marker,
            markersize=3.8,
            linestyle=style_,
            markerfacecolor="white" if split == "ood" else colour,
            markeredgewidth=1.1,
            label="in-distribution" if split == "id" else "out-of-distribution",
        )
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(["0", "1", "2", "3", "4-7", "8+"], fontsize=6.5)
    ax.set_xlabel("Hop distance bin")
    ax.set_ylabel("$S_B$: signed loss mass")
    ax.set_title("Where the task loss is carried", pad=4)
    ax.legend(frameon=False, fontsize=6.0, handlelength=1.4, borderpad=0.1)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.40)

    ax = axes[2]
    for split, colour, marker, style_ in (("id", CARRIAGE, "s", (0, (3.5, 2.2))), ("ood", JACOBIAN, "o", "-")):
        far = {int(k): v for k, v in data[split]["B_far"].items()}
        radii = sorted(far)
        total = far[radii[0]]
        ax.plot(
            radii,
            [far[r] / total for r in radii],
            color=colour,
            marker=marker,
            markersize=3.8,
            linestyle=style_,
            markerfacecolor="white" if split == "ood" else colour,
            markeredgewidth=1.1,
            label="in-distribution" if split == "id" else "out-of-distribution",
        )
    ax.set_xlabel("Radius $r$ (hops)")
    ax.set_ylabel("$B_{\\mathrm{far}}(r)$ / $B_{\\mathrm{far}}(0)$")
    ax.set_xticks(radii)
    ax.set_ylim(0, 1.02)
    ax.set_title("Fraction carried beyond $r$", pad=4)
    ax.legend(frameon=False, fontsize=6.0, handlelength=1.4, borderpad=0.1)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.40)

    out = FIGURES / "fig7_beneficial.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_quantised() -> Path:
    data = load("quantised.json")
    rows = data["results"]
    sweep = [r for r in rows if r["tau"] is not None]
    tau = [r["tau"] for r in sweep]

    def band(row):
        vals = [row["truth"][c]["range"] for c in ("sd", "mad", "gmd")]
        return min(vals), max(vals)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    fig.subplots_adjust(wspace=0.50)

    ax = axes[0]
    lo = [band(r)[0] for r in sweep]
    hi = [band(r)[1] for r in sweep]
    ax.fill_between(tau, lo, hi, color=MUTED, alpha=0.35, linewidth=0, label="ground truth")
    for key, label, colour, marker, style_ in (
        ("carriage_normalised", "finite, donor-avg", CARRIAGE, "s", (0, (3.5, 2.2))),
        ("carriage_perevent", "finite, per event", "#1baf7a", "^", (0, (1.2, 1.6))),
        ("jacobian_ratiomeans", "tangent, pooled", "#eda100", "v", (0, (2.5, 1.5))),
        ("jacobian_meanratio", "tangent, per node (paper)", JACOBIAN, "o", "-"),
    ):
        ax.plot(
            tau, [r[key]["range"] for r in sweep], color=colour, marker=marker, markersize=3.6,
            linestyle=style_, markerfacecolor="white" if key == "jacobian_meanratio" else colour,
            markeredgewidth=1.1, label=label,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Step width $\tau$  (linear $\rightarrow$ step)")
    ax.set_ylabel("Range (hops)")
    ax.set_title("Recovering a known range", pad=4)
    control = next(r for r in rows if r["tau"] is None)
    ax.text(
        0.03, 0.03,
        "linear control: truth and all\narms at {:.3f}".format(control["truth"]["sd"]["range"]),
        transform=ax.transAxes, fontsize=5.8, color=INK, va="bottom",
    )
    ax.legend(frameon=False, fontsize=5.6, handlelength=1.4, borderpad=0.1, loc="upper right")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.36)

    ax = axes[1]
    last = sweep[-1]
    cells = [
        ("tangent\nper node", "jacobian_meanratio", JACOBIAN),
        ("tangent\npooled", "jacobian_ratiomeans", JACOBIAN),
        ("finite\nper event", "carriage_perevent", CARRIAGE),
        ("finite\ndonor-avg", "carriage_normalised", CARRIAGE),
    ]
    positions = np.arange(len(cells))
    values = [last[key]["range"] for _, key, _ in cells]
    errors = np.abs(
        np.array([[last[key]["low"], last[key]["high"]] for _, key, _ in cells]).T - np.array(values)
    )
    ax.bar(
        positions, values, width=0.62,
        color=[c for _, _, c in cells], alpha=0.85, linewidth=0,
        yerr=errors, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#3a3a38"},
    )
    low, high = band(last)
    ax.axhspan(low, high, color=MUTED, alpha=0.35, linewidth=0)
    ax.text(3.4, high, " truth", fontsize=6.0, color=INK, va="bottom", ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels([n for n, _, _ in cells], fontsize=5.6)
    ax.set_ylabel("Range (hops)")
    ax.set_title(r"Both axes matter ($\tau=0.05$)", pad=4)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.36)

    ax = axes[2]
    arms = [
        ("beneficial_sqrt", r"$\sqrt{S_B}$"),
        ("carriage_raw", r"$F_{\mathrm{sens}}$ raw"),
        ("carriage_normalised", r"$F_{\mathrm{sens}}$ norm."),
        ("jacobian_ratiomeans", "tangent pooled"),
        ("jacobian_meanratio", "tangent (paper)"),
    ]
    width = 0.26
    positions = np.arange(len(arms))
    for offset, conv, colour, hatch in (
        (-width, "sd", CARRIAGE, None),
        (0.0, "mad", JACOBIAN, "///"),
        (width, "gmd", "#1baf7a", "..."),
    ):
        mae = [
            float(np.mean([abs(r[key]["range"] - r["truth"][conv]["range"]) for r in rows]))
            for key, _ in arms
        ]
        ax.bar(positions + offset, mae, width=width, color=colour, alpha=0.85,
               linewidth=0, hatch=hatch, label=conv)
    ax.set_xticks(positions)
    ax.set_xticklabels([n for _, n in arms], rotation=32, ha="right", fontsize=5.8)
    ax.set_ylabel("Mean abs. error (hops)")
    ax.set_title("Three spread conventions", pad=4)
    ax.legend(frameon=False, fontsize=5.8, handlelength=1.0, borderpad=0.1, title=None)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.40)

    out = FIGURES / "fig8_quantised.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_gate() -> Path:
    data = load("gate.json")
    rows = data["results"]
    sweep = [r for r in rows if r["tau"] is not None]
    tau = [r["tau"] for r in sweep]
    arms = [
        ("beneficial_sqrt", r"$\sqrt{S_B}$", CARRIAGE, "s", (0, (3.5, 2.2))),
        ("carriage_raw", r"$F_{\mathrm{sens}}$ raw", "#1baf7a", "^", (0, (1.2, 1.6))),
        ("carriage_normalised", r"$F_{\mathrm{sens}}$ norm.", "#eda100", "v", (0, (2.5, 1.5))),
        ("jacobian_pooled", "Jacobian (pooled)", JACOBIAN, "o", "-"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    fig.subplots_adjust(wspace=0.50)

    ax = axes[0]
    ax.fill_between(
        tau,
        [min(r["truth"]["total"]["range"], r["truth"]["total_abs"]["range"]) for r in sweep],
        [max(r["truth"]["total"]["range"], r["truth"]["total_abs"]["range"]) for r in sweep],
        color=MUTED, alpha=0.35, linewidth=0, label="truth (total effect)",
    )
    ax.plot(tau, [r["truth"]["first"]["range"] for r in sweep], color=MUTED,
            linewidth=1.0, linestyle=(0, (1, 2)), label="truth (first order only)")
    for key, label, colour, marker, style_ in arms:
        ax.plot(
            tau, [r[key]["range"]["estimate"] for r in sweep], color=colour, marker=marker,
            markersize=3.6, linestyle=style_,
            markerfacecolor="white" if key == "jacobian_pooled" else colour,
            markeredgewidth=1.1, label=label,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"Gate sharpness: $\tau$ (soft $\rightarrow$ hard)")
    ax.set_ylabel("Range (hops)")
    ax.set_title("Far gate, near value", pad=4)
    ax.legend(frameon=False, fontsize=5.4, handlelength=1.4, borderpad=0.1, loc="upper left")
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.36)

    ax = axes[1]
    last = sweep[-1]
    positions = np.arange(len(arms))
    values = [last[key]["gate_share"]["estimate"] for key, _, _, _, _ in arms]
    errors = np.abs(
        np.array([[last[key]["gate_share"]["low"], last[key]["gate_share"]["high"]]
                  for key, _, _, _, _ in arms]).T - np.array(values)
    )
    ax.bar(positions, values, width=0.6, color=[c for _, _, c, _, _ in arms], alpha=0.85,
           linewidth=0, yerr=errors, error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": "#3a3a38"})
    ax.axhspan(last["truth"]["total_abs"]["gate_share"], last["truth"]["total"]["gate_share"],
               color=MUTED, alpha=0.35, linewidth=0)
    ax.axhline(last["truth"]["first"]["gate_share"], color=MUTED, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.text(len(arms) - 0.4, last["truth"]["first"]["gate_share"], " first order",
            fontsize=5.6, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(positions)
    ax.set_xticklabels([l for _, l, _, _, _ in arms], rotation=32, ha="right", fontsize=5.8)
    ax.set_ylabel("Share of mass at the gate")
    ax.set_title(r"Seeing the gate ($\tau=0.05$)", pad=4)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.40)

    ax = axes[2]
    width = 0.38
    positions = np.arange(len(arms))
    for offset, conv, colour, hatch, name in (
        (-width / 2, "total", CARRIAGE, None, "total (variance)"),
        (width / 2, "total_abs", JACOBIAN, "///", "total (absolute)"),
    ):
        mae = [
            float(np.mean([abs(r[key]["range"]["estimate"] - r["truth"][conv]["range"]) for r in rows]))
            for key, _, _, _, _ in arms
        ]
        ax.bar(positions + offset, mae, width=width, color=colour, alpha=0.85,
               linewidth=0, hatch=hatch, label=name)
    ax.set_xticks(positions)
    ax.set_xticklabels([l for _, l, _, _, _ in arms], rotation=32, ha="right", fontsize=5.8)
    ax.set_ylabel("Mean abs. error (hops)")
    ax.set_title("Two total-effect conventions", pad=4)
    ax.legend(frameon=False, fontsize=5.8, handlelength=1.0, borderpad=0.1)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.40)

    out = FIGURES / "fig9_gate.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def figure_counterflow() -> Path:
    rows = load("counterflow.json")["results"]
    gamma = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    fig.subplots_adjust(wspace=0.50)

    # (a) the tangent collapses with saturation; the finite response never moves.
    ax = axes[0]
    distances = sorted({r["distance"] for r in rows})
    shades = ["#9dc3ee", JACOBIAN, "#17539c"]
    for distance, colour in zip(distances, shades):
        sub = sorted(
            [r for r in rows if r["gamma"] == gamma and r["distance"] == distance and r["tau"] == 1.0],
            key=lambda r: r["kappa"],
        )
        ax.plot(
            [r["kappa"] for r in sub], [r["measured"]["jacobian_range"] for r in sub],
            color=colour, marker="o", markersize=3.6, markerfacecolor="white",
            markeredgewidth=1.1, label=rf"Jacobian, $D={distance}$",
        )
    far = rows[0]["expected"]["F_far"]
    sub = sorted([r for r in rows if r["gamma"] == gamma and r["distance"] == 10 and r["tau"] == 1.0],
                 key=lambda r: r["kappa"])
    ax.plot([r["kappa"] for r in sub], [r["measured"]["F_far"] for r in sub],
            color=CARRIAGE, marker="s", markersize=3.6, linestyle=(0, (3.5, 2.2)),
            label=r"$F_{\mathrm{sens}}$ at $d=D$")
    ax.plot([r["kappa"] for r in sub], [r["measured"]["F_near"] for r in sub],
            color="#1baf7a", marker="^", markersize=3.6, linestyle=(0, (1.2, 1.6)),
            label=r"$F_{\mathrm{sens}}$ at $d=1$")
    ax.set_xscale("log", base=2)
    ax.set_xticks([r["kappa"] for r in sub])
    ax.set_xticklabels([f"{r['kappa']:g}" for r in sub])
    ax.set_xlabel(r"Saturation $\kappa$")
    ax.set_ylabel("Range / response")
    ax.set_title("Saturation hides the far pathway", pad=4)
    ax.legend(frameon=False, fontsize=5.4, handlelength=1.4, borderpad=0.1)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "a", dx=-0.36)

    # (b) counterflow: the local pathway is adverse, the far one beneficial.
    ax = axes[1]
    gammas = sorted({r["gamma"] for r in rows})
    positions = np.arange(len(gammas))
    width = 0.36
    near = [next(r for r in rows if r["gamma"] == g and r["kappa"] == 8.0
                 and r["distance"] == 10 and r["tau"] == 1.0)["measured"]["B_near"] for g in gammas]
    farb = [next(r for r in rows if r["gamma"] == g and r["kappa"] == 8.0
                 and r["distance"] == 10 and r["tau"] == 1.0)["measured"]["B_far"] for g in gammas]
    ax.bar(positions - width / 2, near, width=width, color="#1baf7a", alpha=0.85,
           linewidth=0, label=r"$B$ at $d=1$ (local)")
    ax.bar(positions + width / 2, farb, width=width, color=CARRIAGE, alpha=0.85,
           linewidth=0, label=r"$B$ at $d=D$ (far)")
    ax.plot(positions, np.asarray(near) + np.asarray(farb), linestyle="none", marker="D",
            markersize=4.2, color=INK, label="sum = loss increase")
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"$\\gamma={g:g}$" for g in gammas], fontsize=6.5)
    ax.set_ylabel("Beneficial carriage")
    ax.set_title("Local counterflow", pad=4)
    ax.legend(frameon=False, fontsize=5.8, handlelength=1.0, borderpad=0.1)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "b", dx=-0.40)

    # (c) target alignment moves B alone.
    ax = axes[2]
    sub = sorted([r for r in rows if r["gamma"] == gamma and r["kappa"] == 8.0 and r["distance"] == 10],
                 key=lambda r: r["tau"])
    taus = [r["tau"] for r in sub]
    for key, label, colour, marker, style_ in (
        ("B_far", r"$B$ at $d=D$", CARRIAGE, "s", (0, (3.5, 2.2))),
        ("B_near", r"$B$ at $d=1$", "#1baf7a", "^", (0, (1.2, 1.6))),
        ("F_far", r"$F_{\mathrm{sens}}$ at $d=D$", "#eda100", "v", (0, (2.5, 1.5))),
        ("jacobian_range", "Jacobian range", JACOBIAN, "o", "-"),
    ):
        ax.plot(taus, [r["measured"][key] for r in sub], color=colour, marker=marker,
                markersize=3.6, linestyle=style_,
                markerfacecolor="white" if key == "jacobian_range" else colour,
                markeredgewidth=1.1, label=label)
    ax.axhline(0.0, color=MUTED, linewidth=0.8)
    ax.set_xlabel(r"Target alignment $\tau$  ($y=\tau b$)")
    ax.set_ylabel("Value")
    ax.set_xticks(taus)
    ax.set_title("Only $B$ sees the label", pad=4)
    ax.legend(frameon=False, fontsize=5.6, handlelength=1.4, borderpad=0.1)
    ax.grid(axis="y", alpha=0.9)
    ax.set_axisbelow(True)
    panel_letter(ax, "c", dx=-0.40)

    out = FIGURES / "fig10_counterflow.png"
    fig.savefig(out)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> None:
    style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    builders = [
        ("fig1", figure_replication),
        ("fig1-path", lambda: figure_replication("replication_path.json", "fig1s_replication_path")),
        ("fig2", figure_exactness),
        ("fig3", figure_learned),
        ("fig4", figure_divergence),
        ("fig5", figure_graphlevel),
        ("fig6", figure_realmodel),
        ("fig7", figure_beneficial),
        ("fig8", figure_quantised),
        ("fig9", figure_gate),
        ("fig10", figure_counterflow),
    ]
    for label, builder in builders:
        try:
            print("wrote", builder())
        except FileNotFoundError as error:
            print(f"skipped {label}: {error}")


if __name__ == "__main__":
    main()
