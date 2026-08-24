#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optional plotting helpers for MBXAS spectra. Requires matplotlib
(`pip install pymbxas[plot]`); not part of the core dependencies.
"""


def plot_shakeup_summary(summary, show_probability=True):
    """Plot a `Spectra.get_shakeup_summary()` result: the bare spectrum
    against each intermediate shake-up order, and (optionally) the
    underlying shake-up probability curve.

    Returns (fig, axes): axes is a list of 1 Axes (show_probability=False)
    or 2 (show_probability=True); axes[0] is always the main spectrum
    comparison.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib; install it with pip install pymbxas[plot]."
        )

    energy = summary["energy"]
    spectra = summary["spectra"]
    order_keys = sorted(k for k in spectra if isinstance(k, int))
    max_order = max(order_keys) if order_keys else 0
    has_cross = "cross" in spectra
    plot_keys = order_keys + (["cross"] if has_cross else [])

    if show_probability:
        fig, (ax_main, ax_prob) = plt.subplots(
            2, 1, constrained_layout=True,
            gridspec_kw={"height_ratios": [3, 1]})
        axes = [ax_main, ax_prob]
    else:
        fig, ax_main = plt.subplots(constrained_layout=True)
        axes = [ax_main]

    labels = {0: "no shake-up"}
    labels.update({k: "shakeup_order={}".format(k) for k in order_keys if k > 0})
    styles = {0: dict(color="crimson", lw=1.8)}
    for k in order_keys:
        if k > 0:
            styles[k] = dict(lw=1.6, ls="--" if k == 1 else ":")
    if has_cross:
        labels["cross"] = "cross-spin + shake-up"
        styles["cross"] = dict(color="teal", lw=1.6, ls="-.")

    for k in plot_keys:
        ax_main.plot(energy, spectra[k], label=labels[k], **styles[k])
    ax_main.set_xlim(energy[0], energy[-1])
    ax_main.set_ylim(bottom=0)
    ax_main.set_ylabel("Intensity (arb. units)")
    ax_main.legend(frameon=False)

    if show_probability:
        prob_e, prob_curve, prob_orders = summary["probability"]
        ax_prob.plot(prob_e, prob_curve, color="darkorange", lw=1.6)
        ax_prob.set_xlim(prob_e[0], prob_e[-1])
        ax_prob.set_ylim(bottom=0)
        ax_prob.set_xlabel(r"$\Delta E$ (eV)")
        ax_prob.set_ylabel("Shake-up\nprobability")
        ax_prob.set_title(
            "Valence shake-up probability spectrum (orders {})".format(prob_orders),
            fontsize=10)
    else:
        ax_main.set_xlabel("Energy (eV)")

    return fig, axes
