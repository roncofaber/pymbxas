"""Optional Matplotlib helpers for MBXAS post-processing.

Matplotlib is imported only when a plotting function is called, keeping it an
optional dependency of the numerical package.
"""

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Plotting requires matplotlib; install it with "
            "pip install pymbxas[plot].") from None
    return plt


def _order_colors(plt, orders):
    palette = plt.get_cmap("tab10").colors
    return {
        order: palette[index % len(palette)]
        for index, order in enumerate(sorted(orders))
    }


def plot_mbxas_decomposition(
        decomposition, show_probability=True, *, show_resolved=False,
        show_cumulative=False, figsize=None):
    """Plot a total MBXAS spectrum and its resolved contributions.

    Parameters
    ----------
    decomposition
        Mapping returned by :meth:`Spectra.get_mbxas_decomposition`.
    show_probability
        Add the determinant-overlap probability panel. Retained as a
        positional argument for backward compatibility.
    show_resolved
        Add a panel separating each higher f order into its shake-up and
        shake-down parts.
    show_cumulative
        Add dashed intermediate cumulative spectra to the main panel. The
        highest cumulative curve is already shown as ``Total``; f1 is already
        an individual contribution, so only genuinely distinct intermediate
        curves are added.
    figsize
        Optional Matplotlib figure size.

    Returns
    -------
    (figure, axes)
        ``axes`` is an ordered list containing the main spectrum panel,
        followed by the optional resolved and probability panels.
    """
    plt = _pyplot()
    energy = decomposition["energy"]
    contributions = decomposition["contributions"]
    cumulative = decomposition["cumulative"]
    resolved = decomposition["decomposition"]
    orders = sorted(contributions)
    if not orders:
        raise ValueError("decomposition contains no f-order contributions")
    if show_resolved and not resolved:
        raise ValueError(
            "show_resolved requires a decomposition through at least f2")

    panel_count = 1 + int(show_resolved) + int(show_probability)
    if figsize is None:
        figsize = (7.4, 3.8 + 2.2 * (panel_count - 1))
    heights = [3.0]
    if show_resolved:
        heights.append(1.8)
    if show_probability:
        heights.append(1.3)
    figure, axes_array = plt.subplots(
        panel_count, 1, figsize=figsize, squeeze=False,
        constrained_layout=True, gridspec_kw={"height_ratios": heights})
    axes = list(axes_array[:, 0])
    main = axes[0]
    colors = _order_colors(plt, orders)

    main.plot(
        energy, decomposition["total"], color="black", lw=2.0,
        label=f"Total through f{orders[-1]}")
    for order in orders:
        main.plot(
            energy, contributions[order], color=colors[order], lw=1.6,
            label=f"f{order} contribution")
    if show_cumulative:
        for order in orders:
            if order <= 1 or order == orders[-1]:
                continue
            main.plot(
                energy, cumulative[order], color=colors[order], lw=1.3,
                ls="--", alpha=0.8, label=f"Cumulative through f{order}")
    main.set(
        xlim=(energy[0], energy[-1]), ylim=(0, None),
        xlabel="Photon energy (eV)", ylabel="Intensity (arb. units)")
    main.legend(frameon=False)

    next_axis = 1
    if show_resolved:
        resolved_axis = axes[next_axis]
        next_axis += 1
        for order in sorted(resolved):
            color = colors[order]
            resolved_axis.plot(
                energy, resolved[order]["shakeup"], color=color, lw=1.5,
                label=f"f{order} shake-up")
            resolved_axis.plot(
                energy, resolved[order]["shakedown"], color=color, lw=1.5,
                ls="--", label=f"f{order} shake-down")
        resolved_axis.set(
            xlim=(energy[0], energy[-1]), ylim=(0, None),
            xlabel="Photon energy (eV)",
            ylabel="Resolved intensity\n(arb. units)")
        resolved_axis.legend(frameon=False, ncols=2)

    if show_probability:
        probability_axis = axes[next_axis]
        probability_energy, probability, probability_orders = (
            decomposition["probability"])
        probability_axis.plot(
            probability_energy, probability, color="darkorange", lw=1.6)
        probability_axis.set(
            xlim=(probability_energy[0], probability_energy[-1]),
            ylim=(0, None), xlabel=r"$\Delta E$ (eV)",
            ylabel="Overlap\nweight")
        probability_axis.set_title(
            "Valence determinant-overlap distribution "
            f"(orders {probability_orders})", fontsize=10)

    return figure, axes


def plot_orbital_rearrangement(
        rearrangement, *, show_indices=False, show_dos=False,
        dos_sigma=0.25, figsize=(10.0, 6.0)):
    """Plot GS and FCH orbital levels with overlap-based identity lines.

    Parameters
    ----------
    rearrangement
        Mapping returned by :meth:`Spectra.get_orbital_rearrangement`.
    show_indices
        Annotate each selected level with its zero-based MO index.
    show_dos
        Draw outward-facing broadened orbital-level densities in the unused
        margins beside the GS and FCH level columns.
    dos_sigma
        Gaussian broadening in eV for the orbital-level densities. GS and FCH
        share one density scale within each spin panel.
    figsize
        Matplotlib figure size.

    Returns
    -------
    (figure, axes)
        A figure and two axes ordered alpha, beta.
    """
    plt = _pyplot()
    from matplotlib.lines import Line2D

    channels = rearrangement["channels"]
    if len(channels) != 2:
        raise ValueError("orbital rearrangement requires alpha and beta channels")
    if dos_sigma <= 0:
        raise ValueError("dos_sigma must be positive")

    selected_energies = []
    for channel in channels:
        selected_energies.extend(channel["gs_energy"][
            channel["selected_gs"]].tolist())
        selected_energies.extend(channel["fch_energy"][
            channel["selected_fch"]].tolist())
    if not selected_energies:
        raise ValueError("No orbitals fall inside the selected energy window")
    selected_min = min(selected_energies)
    selected_max = max(selected_energies)
    requested_window = rearrangement.get("energy_window")
    if requested_window is None:
        span = max(selected_max - selected_min, 1.0)
        margin = 0.05 * span
        energy_min = selected_min - margin
        energy_max = selected_max + margin
    else:
        energy_min, energy_max = map(float, requested_window)
        # ``include_core=True`` can deliberately select a level outside the
        # frontier window. Expand only the side containing that extra level;
        # ordinary fixed-window plots retain identical limits across sites.
        expanded_span = max(
            max(energy_max, selected_max) - min(energy_min, selected_min), 1.0)
        outside_margin = 0.02 * expanded_span
        if selected_min < energy_min:
            energy_min = selected_min - outside_margin
        if selected_max > energy_max:
            energy_max = selected_max + outside_margin
    figure, axes_array = plt.subplots(
        1, 2, figsize=figsize, sharey=True)
    figure.subplots_adjust(
        left=0.09, right=0.98, bottom=0.22, top=0.84, wspace=0.03)
    axes = list(axes_array)
    state_x = {"gs": 0.0, "fch": 1.0}
    bar_width = 0.45
    dos_width = 0.23
    frontier_colors = {"homo": "#d28e00", "lumo": "#008b95"}

    def level_style(channel, state, index):
        occupation = channel[f"{state}_occupation"][index]
        color = "#303030" if occupation > 0.5 else "#a8a8a8"
        linewidth = 1.0 if occupation > 0.5 else 0.55
        zorder = 3
        if index == channel[f"{state}_homo"]:
            color = frontier_colors["homo"]
            linewidth = 2.2
            zorder = 4
        if index == channel[f"{state}_lumo"]:
            color = frontier_colors["lumo"]
            linewidth = 2.2
            zorder = 4
        core_key = "core_gs" if state == "gs" else "core_fch"
        if index == channel[core_key]:
            color = "crimson"
            linewidth = 2.4
            zorder = 5
        return color, linewidth, zorder

    for spin, (axis, channel) in enumerate(zip(axes, channels)):
        if show_dos:
            density_energy = np.linspace(
                energy_min, energy_max, 600)
            densities = {}
            for state, selection_key in (("gs", "selected_gs"),
                                         ("fch", "selected_fch")):
                levels = channel[f"{state}_energy"][channel[selection_key]]
                delta = ((density_energy[:, None] - levels[None, :])
                         / dos_sigma)
                densities[state] = np.exp(-0.5 * delta**2).sum(axis=1)
            density_scale = max(
                float(np.max(density)) for density in densities.values())
            if density_scale > 0:
                for state, direction in (("gs", -1.0), ("fch", 1.0)):
                    baseline = (state_x[state]
                                + direction * bar_width / 2)
                    profile = (baseline + direction * dos_width
                               * densities[state] / density_scale)
                    axis.fill_betweenx(
                        density_energy, baseline, profile,
                        color="#6f6f6f", alpha=0.07, linewidth=0,
                        zorder=0)
                    axis.plot(
                        profile, density_energy, color="#6f6f6f",
                        lw=0.65, alpha=0.5, zorder=0.5)

        matches = channel["matches"]
        for gs_index, fch_index, weight in zip(
                matches["gs_index"], matches["fch_index"],
                matches["overlap"]):
            occupation_changed = (
                (channel["gs_occupation"][gs_index] > 0.5)
                != (channel["fch_occupation"][fch_index] > 0.5))
            if occupation_changed:
                color = "crimson"
                linewidth = 0.8 + 1.0 * weight
                alpha = 0.25 + 0.65 * weight
            else:
                color = "#777777"
                linewidth = 0.35 + 0.65 * weight
                alpha = 0.025 + 0.30 * weight
            axis.plot(
                [state_x["gs"] + bar_width / 2,
                 state_x["fch"] - bar_width / 2],
                [channel["gs_energy"][gs_index],
                 channel["fch_energy"][fch_index]],
                ls="--", lw=linewidth, color=color, alpha=alpha, zorder=1)

        for state, selection_key in (("gs", "selected_gs"),
                                     ("fch", "selected_fch")):
            energies = channel[f"{state}_energy"]
            indices = np.flatnonzero(channel[selection_key])
            for index in indices:
                color, linewidth, zorder = level_style(
                    channel, state, index)
                axis.plot(
                    [state_x[state] - bar_width / 2,
                     state_x[state] + bar_width / 2],
                    [energies[index], energies[index]],
                    color=color, lw=linewidth, solid_capstyle="butt",
                    zorder=zorder)
                if show_indices:
                    offset = -bar_width / 2 - 0.035 if state == "gs" else (
                        bar_width / 2 + 0.035)
                    axis.text(
                        state_x[state] + offset, energies[index], str(index),
                        ha="right" if state == "gs" else "left",
                        va="center", fontsize=7, color="#444444")

        spin_label = r"$\alpha$" if spin == 0 else r"$\beta$"
        axis.set(
            xlim=(-0.55, 1.55), ylim=(energy_min, energy_max),
            xticks=[0.0, 1.0], xticklabels=["GS", "FCH"],
            title=f"{spin_label} spin — {channel['role']}")
        axis.tick_params(axis="x", length=0)
        axis.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
        axis.set_axisbelow(True)

    axes[0].set_ylabel(
        f"Orbital energy relative to {rearrangement['reference_label']} (eV)")
    title = "GS → FCH orbital rearrangement"
    if rearrangement["site"] is not None:
        title += f" — {rearrangement['site']}"
    figure.suptitle(title, y=0.95)
    legend_handles = [
        Line2D([0], [0], color="#303030", lw=1.0, label="Occupied"),
        Line2D([0], [0], color="#a8a8a8", lw=0.55, label="Unoccupied"),
        Line2D([0], [0], color=frontier_colors["homo"], lw=2.2,
               label="HOMO"),
        Line2D([0], [0], color=frontier_colors["lumo"], lw=2.2,
               label="LUMO"),
        Line2D([0], [0], color="crimson", lw=2.4,
               label="Excited core / core hole"),
        Line2D([0], [0], color="#777777", ls="--", label="MO-overlap match"),
        Line2D([0], [0], color="crimson", ls="--",
               label="Occupation-changing match"),
    ]
    figure.legend(
        handles=legend_handles, loc="lower center", ncols=4,
        bbox_to_anchor=(0.5, 0.015), frameon=False)
    return figure, axes
