"""Shared paper-quality matplotlib style (import to apply)."""
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11, "figure.titlesize": 12,
    "legend.fontsize": 9, "legend.frameon": False,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "lines.linewidth": 1.9, "lines.markersize": 5,
    "axes.prop_cycle": mpl.cycler(color=[
        "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]),
})
PALETTE = {"firstord": "#d62728", "direct_marginal": "#1f77b4", "direct_crn": "#17becf",
           "direct_donor": "#2ca02c", "perturb_small": "#9467bd", "perturb_large": "#8c564b",
           "ig": "#ff7f0e", "eg": "#bcbd22"}
