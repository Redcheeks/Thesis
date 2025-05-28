import os
from matplotlib.figure import Figure, Axes
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Type
from neuron import Neuron, NeuronFactory
from simulation.models.LIF_model_SIMPLE import LIF_SIMPLE
from simulation.models.LIF_model1 import LIF_Model1
from simulation.models.LIF_model2_v3 import LIF_Model2v3
from simulation.models.LIF_model3 import LIF_Model3
from simulation.models.LIF_model3_v2 import LIF_Model3v2
from brokenaxes import brokenaxes  # pip install brokenaxes
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Script for Frequency - Current (DC) plots.
Instructions to use: 
- Choose parameters below
- In _main() - Make sure correct model is uncommented 

For details on how plots are made, see the relevant method

"""
### ------------- Simulation PARAMETERS ------------- ###
T = 500  # Simulation Time [ms]
DT = 0.02  # Time step in [ms]
neuron_pool_size = 300  # Total number of Neurons in the pool

## ------ Neurons to be modelled & plotted. ------ ##
NEURON_INDEXES: List[int] = [1, 30, 50, 100, 150, 200]
# choose some neurons to plot! Indexes must be within {1 : neuron_pool_size}

### ------ Cortical input Simulation PARAMETERS ------ ###

MIN_I = 2  # Min input current (nA)
MAX_I = 35  # Max input current (nA)
I_SAMPLES = 2 * (MAX_I - MIN_I)  # How many current levels to run for.

## ------------- Plotting text-sizes ------------- ##
import matplotlib.pylab as pylab

params = {
    "legend.fontsize": "medium",
    "axes.labelsize": "large",
    "axes.titlesize": "x-large",
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "figure.titlesize": "xx-large",
}
pylab.rcParams.update(params)


def FI_curves(neuron, model, fig: Figure):

    cur = np.linspace(MIN_I, MAX_I, I_SAMPLES, dtype=float)
    freq_mean = np.zeros(len(cur))
    freq_peak = np.zeros(len(cur))

    for i, Iinj in enumerate(cur):
        current = Iinj * np.ones(int(T / DT))
        v, sp = model.simulate_neuron(T, DT, neuron, current)[:2]
        # only interested in 2nd return value
        if len(sp) > 1:
            freq_mean[i] = np.median((1 / np.diff(sp)) * 1e3)
            freq_peak[i] = np.max((1 / np.diff(sp)) * 1e3)

    if model == LIF_SIMPLE:  # Only one plot needed.

        plt.plot(cur, freq_mean, "o:", linewidth=0.5)
        plt.xlabel("Current [nA]")
        plt.ylabel("Frequency [Hz]")
        plt.title(f"Median frequencies")
        plt.xlim((MIN_I, MAX_I))
    elif model == LIF_Model1:
        ax, ax2, ax4 = fig.get_axes()

        ax.plot(cur, freq_mean, "o:", linewidth=0.5)
        ax.set_ylim(-1, 50)

        # PEAK F-I
        ax2.plot(cur, freq_peak, "o:", linewidth=0.5)

        ax4.plot(cur, freq_peak, "o:", linewidth=0.5)

    else:  # plot average and peak

        ax, ax2, ax3, ax4 = fig.get_axes()

        # AVERAGE F-I

        ax.plot(cur, freq_mean, "o:", linewidth=0.5)

        ax3.plot(cur, freq_mean, "o:", linewidth=0.5)

        # PEAK F-I
        ax2.plot(cur, freq_peak, "o:", linewidth=0.5)

        ax4.plot(cur, freq_peak, "o:", linewidth=0.5)

    #     plt.subplot(2, 1, 1)
    #     # if max(freq_mean) > 60:
    #     #     bax = brokenaxes(
    #     #         ylims=((0, 50), (min(freq_mean[freq_mean > 50]), max(freq_mean))),
    #     #         hspace=0.05,
    #     #     )
    #     #     bax.plot(cur, freq_mean)
    #     #     bax.set_label("Current [nA]")
    #     #     bax.set_ylabel("Frequency [Hz]")
    #     # else:
    #     plt.plot(cur, freq_mean, ".")
    #     plt.xlabel("Current [nA]")
    #     plt.ylabel("Frequency [Hz]")
    #     plt.title(f"Average frequencies")
    #     plt.xlim((MIN_I, MAX_I))

    #     plt.subplot(2, 1, 2)
    #     # if max(freq_peak) > 60:
    #     #     bax = brokenaxes(
    #     #         ylims=((0, 50), (min(freq_peak[freq_peak > 50]), max(freq_peak))),
    #     #         hspace=0.05,
    #     #     )
    #     #     bax.plot(cur, freq_peak)
    #     #     bax.set_label("Current [nA]")
    #     #     bax.set_ylabel("Frequency [Hz]")
    #     # else:
    #     plt.plot(cur, freq_peak, ".")
    #     plt.xlabel("Current [nA]")
    #     plt.ylabel("Frequency [Hz]")
    #     plt.title(f"Peak frequencies")
    #     plt.xlim((MIN_I, MAX_I))
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.5, wspace=0.5)


def split_ax_formatting(UL: Axes, UU: Axes, LL: Axes, LU: Axes):
    # Upper subplot - Lower half -
    UL.set_xlabel("Current [nA]")
    # UL.set_ylabel("Frequency [Hz]")
    UL.set_xlim((MIN_I, MAX_I))
    UL.spines["top"].set_visible(False)

    # Upper subplot - Upper half
    UU.set_title(f"Median frequencies")
    UU.set_xlim((MIN_I, MAX_I))
    UU.spines["bottom"].set_visible(False)
    UU.tick_params(bottom=False, labelbottom=False)

    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=UU.transAxes, color="k", clip_on=False)
    UU.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    UU.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=UL.transAxes)  # switch to the bottom axes
    UL.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    UL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    # Lower subplot - Lower half -
    LL.set_xlabel("Current [nA]")
    # LL.set_ylabel("Frequency [Hz]")
    LL.set_xlim((MIN_I, MAX_I))
    LL.spines["top"].set_visible(False)

    # Lower subplot - Upper half
    LU.set_title(f"Peak frequencies")
    LU.set_xlim((MIN_I, MAX_I))
    LU.spines["bottom"].set_visible(False)
    LU.tick_params(bottom=False, labelbottom=False)

    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=LU.transAxes, color="k", clip_on=False)
    LU.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    LU.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=LL.transAxes)  # switch to the bottom axes
    LL.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    LL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    LL.grid(True, "major", "both", linestyle="--", linewidth=0.5)
    LU.grid(True, "major", "both", linestyle="--", linewidth=0.5)
    UL.grid(True, "major", "both", linestyle="--", linewidth=0.5)
    UU.grid(True, "major", "both", linestyle="--", linewidth=0.5)


def split_ax_formatting_3plots(U: Axes, LL: Axes, LU: Axes):
    # Upper subplot - Lower half -
    U.set_xlabel("Current [nA]")
    # UL.set_ylabel("Frequency [Hz]")
    U.set_xlim((MIN_I, MAX_I))
    # U.spines["top"].set_visible(False)

    # Upper subplot - Upper half
    U.set_title(f"Median frequencies")
    U.set_xlim((MIN_I, MAX_I))
    # UU.spines["bottom"].set_visible(False)
    # UU.tick_params(bottom=False, labelbottom=False)

    # Lower subplot - Lower half -
    LL.set_xlabel("Current [nA]")
    # LL.set_ylabel("Frequency [Hz]")
    LL.set_xlim((MIN_I, MAX_I))
    LL.spines["top"].set_visible(False)

    # Lower subplot - Upper half
    LU.set_title(f"Peak frequencies")
    LU.set_xlim((MIN_I, MAX_I))
    LU.spines["bottom"].set_visible(False)
    LU.tick_params(bottom=False, labelbottom=False)

    # From https://matplotlib.org/examples/pylab_examples/broken_axis.html
    d = 0.015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=LU.transAxes, color="k", clip_on=False)
    LU.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    LU.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

    kwargs.update(transform=LL.transAxes)  # switch to the bottom axes
    LL.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    LL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    LL.grid(True, "major", "both", linestyle="--", linewidth=0.5)
    LU.grid(True, "major", "both", linestyle="--", linewidth=0.5)
    U.grid(True, "major", "both", linestyle="--", linewidth=0.5)


def _main():
    # Create neurons with NEURONFACTORY class (Generates list of neurons)

    neurons = NeuronFactory.create_neuron_subpool(
        NEURON_INDEXES, distribution=True, number_of_neurons=neuron_pool_size
    )

    ## -------- F-I Curve for basic LIF -------- ##

    model = LIF_SIMPLE
    fig = plt.figure(figsize=(10, 5))
    plt.grid(True, "major", "both", linestyle="--", linewidth=0.5)

    for it, neuron in enumerate(neurons):  # for each neuron
        FI_curves(neuron, model, fig)

    plt.suptitle(
        f"F-I curves for Simple LIF model, DC input {MIN_I}-{MAX_I} [nA], Simulation time {T} [ms]"
    )
    plt.legend([f"Neuron {i}" for i in NEURON_INDEXES])

    os.makedirs("figures/F-I", exist_ok=True)
    plt.savefig(
        f"figures/F-I/FI_simpleLIF.png",
    )
    print("Simple LIF Figure DONE!")

    ## -------- F-I Curve for model 1 -------- ##

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 8))
    # divider = make_axes_locatable(axes[0])
    # ax2 = divider.new_vertical(size="100%", pad=0.1)
    # fig.add_axes(ax2)

    divider = make_axes_locatable(axes[1])
    ax3 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax3)

    model = LIF_Model1
    plt.suptitle(
        f"F-I curves for LIF model 1, DC input {MIN_I}-{MAX_I} [nA], Simulation time {T} [ms] "
    )
    fig.supylabel("Frequency [Hz]")

    for it, neuron in enumerate(neurons):  # for each neuron
        FI_curves(neuron, model, fig)

    [ax, ax2, ax4] = fig.get_axes()
    split_ax_formatting_3plots(ax, ax2, ax4)  # ax3 is the part split from ax?...
    # ax.set_ylim(-1, 15)  # average lower axes
    # ax3.set_ylim(30, 50)  # average upper axes

    ax2.set_ylim(-1, 15)  # peak lower axes
    ax4.set_ylim(125, 180)  # peak upper axes

    ax.set_ylim(-1, 25)

    plt.legend([f"Neuron {i}" for i in NEURON_INDEXES])
    os.makedirs("figures/F-I", exist_ok=True)
    plt.savefig(
        f"figures/F-I/FI_LIF1.png",
    )
    print("LIF1 Figure DONE!")

    ## -------- F-I Curve for model 2 -------- ##

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    divider = make_axes_locatable(axes[0])
    ax2 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax2)

    divider = make_axes_locatable(axes[1])
    ax3 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax3)

    model = LIF_Model2v3
    plt.suptitle(
        f"F-I curves for LIF model 2,  DC input {MIN_I}-{MAX_I} [nA], Simulation time {T} [ms]"
    )
    fig.supylabel("Frequency [Hz]")

    for it, neuron in enumerate(neurons):  # for each neuron
        FI_curves(neuron, model, fig)
    [ax, ax2, ax3, ax4] = fig.get_axes()
    split_ax_formatting(ax, ax3, ax2, ax4)  # ax3 is the part split from ax?...
    ax.set_ylim(-1, 50)  # average lower axes
    ax3.set_ylim(50, 250)  # average upper axes

    ax2.set_ylim(-1, 50)  # peak lower axes
    ax4.set_ylim(100, 250)  # peak upper axes

    plt.legend([f"Neuron {i}" for i in NEURON_INDEXES])

    os.makedirs("figures/F-I", exist_ok=True)
    plt.savefig(
        f"figures/F-I/FI_LIF2.png",
    )
    print("LIF2 Figure DONE!")
    ## -------- F-I Curve for model 3 -------- ##

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    divider = make_axes_locatable(axes[0])
    ax2 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax2)

    divider = make_axes_locatable(axes[1])
    ax3 = divider.new_vertical(size="100%", pad=0.1)
    fig.add_axes(ax3)

    model = LIF_Model3
    plt.suptitle(
        f"F-I curves for LIF model 3, DC input {MIN_I}-{MAX_I} [nA], Simulation time {T} [ms] "
    )
    fig.supylabel("Frequency [Hz]")

    for it, neuron in enumerate(neurons):  # for each neuron
        FI_curves(neuron, model, fig)

    [ax, ax2, ax3, ax4] = fig.get_axes()
    split_ax_formatting(ax, ax3, ax2, ax4)  # ax3 is the part split from ax?...
    ax.set_ylim(-1, 30)  # average lower axes
    ax3.set_ylim(50, 250)  # average upper axes

    ax2.set_ylim(-1, 30)  # peak lower axes
    ax4.set_ylim(100, 250)  # peak upper axes
    plt.legend([f"Neuron {i}" for i in NEURON_INDEXES])

    os.makedirs("figures/F-I", exist_ok=True)
    plt.savefig(
        f"figures/F-I/FI_LIF3.png",
    )
    print("LIF3 Figure DONE!")
    ## -------- F-I Curve for model 3v2 -------- ##

    # plt.figure(figsize=(10, 5))
    # model = LIF_Model3v2
    # plt.suptitle("F-I curves for LIF model 3v2 ")

    # for it, neuron in enumerate(neurons):  # for each neuron
    #     FI_curves(neuron, model)

    # plt.legend([f"Neuron {i}" for i in NEURON_INDEXES])

    # os.makedirs("figures/F-I", exist_ok=True)
    # plt.savefig(
    #     f"figures/F-I/FI_LIF3v2.png",
    # )

    plt.show()


if __name__ == "__main__":
    _main()
