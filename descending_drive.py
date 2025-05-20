import numpy as np
import os
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns  # you may need to run:     conda install seaborn -c conda-forge and maybe update your environment


## -------- TEST Plotting Parameters ----------

T_SIM = 30e3  # Total time in ms
DT_SIM = 0.1  # Time step in ms
MN_POOL_SIZE = 300  # Number of motor neurons
N_CLUST = 3  # Number of clusters
INPUT_AMP = 9  # Max input current (nA)
CCOV = 14  # Common noise CoV (%)
ICOV = 5  # Independent noise CoV (%)

# Note: Signal shape and frequency are to be set in the plot_CI()


def cortical_input(
    n_mn, n_clust, max_I, T_dur, dt, CCoV=20, ICoV=5, mean_shape="trapezoid", freq=0.2
):
    """
    Generates a signal simulating the cortical input to the motorneurons.

    Args:
        n_mn (float) : Number of motorneuron.
        n_clust (float) : Number of cluster.
        max_I () : Maximum current (nA)
        T_dur : Duration / Simulation Time [ms]
        dt : Time step in [ms]
        mean_shape (string): Underlying wave-shape
        CCov : Common noise CoV (%)
        ICov : Independent noise CoV (%)

    Returns:
        np.array: Current vector.
    """

    # CCoV = 20; % percent of mean
    # ICoV = 0.25*CCoV; % percent of mean

    ext_sig = 20
    # Extra 20 seconds for transient removal
    T_dur_ext = T_dur + ext_sig
    # Total duration including transient
    fs = 1 / (dt * 1e-3)

    freq = freq * 1e-3  # hz to 1/ms

    ##The ramp time is based on the experimental data in this thesis.
    if mean_shape == "DC":
        zero_time = 0
        ramp_time = 0
    else:
        ramp_time = 5e3  # 5seconds long ramp
        zero_time = 2e3  # 2 seconds zero time

    # If T_dur is to short, add time for ramps in the reevant shapes
    if T_dur < 2 * ramp_time:
        if mean_shape == ("trapezoid" or "step"):
            T_dur += 2 * ramp_time
        elif mean_shape == "step-sinusoid":
            T_dur += ramp_time

    time = np.arange(
        0, T_dur + 2 * zero_time, dt
    )  # pad total time to add zero input start and end.
    if mean_shape == "DC":
        # Compute the number of samples for each phase
        zero_len = int(round(zero_time) / dt)  # Zero start and end
        ramp_len = int(round(ramp_time) / dt)  # Ramp-up and ramp-downz
        hold_len = int(round(T_dur - 2 * ramp_time) / dt)  # + 1  # Hold phase

        # Create each section
        zero_segment = np.zeros(zero_len)
        ramp_up = np.linspace(0, max_I, ramp_len)
        hold_segment = np.full(hold_len, max_I)
        ramp_down = np.flip(np.linspace(0, max_I, ramp_len))

        mean_drive = np.concatenate(
            [zero_segment, ramp_up, hold_segment, ramp_down, zero_segment]
        )

    elif mean_shape == "trapezoid":
        # Compute the number of samples for each phase
        zero_len = int(round(zero_time) / dt)  # Zero start and end
        ramp_len = int(round(ramp_time) / dt)  # Ramp-up and ramp-down
        hold_len = int(round(T_dur - 2 * ramp_time) / dt)  # + 1  # Hold phase

        # Create each section
        zero_segment = np.zeros(zero_len)
        ramp_up = np.linspace(0, max_I, ramp_len)
        hold_segment = np.full(hold_len, max_I)
        ramp_down = np.flip(np.linspace(0, max_I, ramp_len))

        mean_drive = np.concatenate(
            [zero_segment, ramp_up, hold_segment, ramp_down, zero_segment]
        )

    elif mean_shape == "triangular":
        # Compute the number of samples for each phase
        zero_len = int(round(zero_time) / dt)  # Zero start and end
        ramp_len = int(round(0.5 * T_dur) / dt)  # Ramp-up and ramp-down

        # Create each section
        zero_segment = np.zeros(zero_len)
        ramp_up = np.linspace(0, max_I, ramp_len)
        ramp_down = np.flip(np.linspace(0, max_I, ramp_len))

        mean_drive = np.concatenate([zero_segment, ramp_up, ramp_down, zero_segment])

    elif mean_shape == "sinusoid.hz":

        zero_len = int(round(zero_time) / dt)  # Zero start and end
        time_sin = np.arange(0, T_dur, dt)  # Time vector

        # Generate the sinusoidal wave
        sine_wave = np.sin(-0.5 * np.pi + 2 * np.pi * freq * time_sin)
        zero_segment = np.zeros(zero_len)

        # Normalize to range [0, max_I]
        mean_drive = np.concatenate(
            [zero_segment, (max_I * (sine_wave + 1) / 2), zero_segment]
        )

    elif mean_shape == "step-sinusoid":
        # Compute the number of samples for each phase
        zero_len = int(round(zero_time) / dt)  # Zero start and end
        ramp_len = int(round(ramp_time) / dt)  # Ramp-up
        hold_len = int(round((T_dur - ramp_time) / 2) / dt)  # Hold phase
        sine_len = int(round((T_dur - ramp_time) / 2) / dt)  # Sine Phase

        # Create each section
        zero_segment = np.zeros(zero_len)
        ramp_up = np.linspace(0, max_I, ramp_len)
        hold_segment = np.full(hold_len, max_I)

        # Generate the sinusoidal wave
        time_sin = np.arange(0, sine_len)  # Time vector
        sine_wave = np.sin(-0.5 * np.pi + 2 * np.pi * freq * time_sin)

        # Normalize to range []
        sine_segment = max_I + (max_I / 3 * (sine_wave))
        mean_drive = np.concatenate(
            [zero_segment, ramp_up, hold_segment, sine_segment, zero_segment]
        )

    else:  # elif mean_shape == "step":

        # Compute the number of samples for each phase
        zero_len = int(round(zero_time) / dt)  # Zero start and end
        ramp_len = int(round((2 / 12) * T_dur) / dt)  # Ramp-up
        hold_len = int(round((4 / 12) * T_dur) / dt)  # Hold phase

        # Create each section
        zero_segment = np.zeros(zero_len)
        ramp_up_1 = np.linspace(0, 0.5 * max_I, ramp_len)
        hold_segment_1 = np.full(hold_len, 0.5 * max_I)
        ramp_up_2 = np.linspace(0.5 * max_I, max_I, ramp_len)
        hold_segment_2 = np.full(hold_len, max_I)

        mean_drive = np.concatenate(
            [
                zero_segment,
                ramp_up_1,
                hold_segment_1,
                ramp_up_2,
                hold_segment_2,
                zero_segment,
            ]
        )

    mean_CI = max_I  # max(mean_drive)

    # Filter design
    b_low, a_low = signal.butter(2, 100 / fs * 2, "low")  # Low-pass filter
    b_band, a_band = signal.butter(
        2, [15 / fs * 2, 35 / fs * 2], "bandpass"
    )  # Band-pass filter

    # Generate common noise - one set of noise per cluster
    gNoise = np.random.randn(n_clust, len(time))  # White noise
    commonNoise = signal.filtfilt(b_band, a_band, gNoise, axis=1)  # Band-pass filter

    # Scale common noise correctly
    std_common = np.std(
        commonNoise, axis=1, keepdims=True
    )  # Keep shape as (n_clust, 1)
    drive_common = (commonNoise / std_common) * (CCoV / 100 * mean_CI)

    # Structured Cluster Assignment
    avg_mn_per_clust = n_mn // n_clust  # Neurons per cluster (integer division)
    drive_com_new = np.zeros((n_mn, drive_common.shape[1]))  # Initialize matrix

    # Assign neurons sequentially to clusters
    for cluster_idx in range(n_clust):
        start_idxx = cluster_idx * avg_mn_per_clust
        end_idxx = (
            (cluster_idx + 1) * avg_mn_per_clust if cluster_idx < n_clust - 1 else n_mn
        )  # Ensure all neurons are assigned
        drive_com_new[start_idxx:end_idxx, :] = drive_common[
            cluster_idx, :
        ]  # Assign cluster noise

    # Generate independent noise for each motor neuron
    gNoise = np.random.randn(n_mn, len(time))  # White noise
    indNoise = signal.filtfilt(b_low, a_low, gNoise, axis=1)  # Low-pass filter

    # Scale independent noise
    drive_ind = (indNoise / np.std(indNoise, axis=0, keepdims=True)) * (
        ICoV / 100 * mean_CI
    )

    # Combine mean drive, common drive, and independent noise
    CI = (
        mean_drive[:, np.newaxis] + drive_com_new.T + drive_ind.T
    )  # Final cortical input matrix
    CI = np.clip(CI, 0, None)
    return CI


def plot_CI():

    CI_sinus = cortical_input(
        MN_POOL_SIZE,
        N_CLUST,
        INPUT_AMP,
        T_SIM,
        DT_SIM,
        CCOV,
        ICOV,
        "step-sinusoid",
        freq=0.2,
    )

    CI_trap = cortical_input(
        MN_POOL_SIZE,
        N_CLUST,
        INPUT_AMP,
        T_SIM,
        DT_SIM,
        CCOV,
        ICOV,
        "trapezoid",
    )

    plt.figure(figsize=(18, 10))
    plt.tight_layout()
    time = np.linspace(0, T_SIM, CI_trap.shape[0])

    plt.subplot(2, 1, 1)
    plt.plot(time * 1e-3, CI_trap[:, 50], label=f"Neuron {50}")
    plt.axhline(INPUT_AMP, color="r", ls="--", alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (nA)")
    plt.title(f"Trapezoid input")

    plt.subplot(2, 1, 2)
    plt.plot(time * 1e-3, CI_sinus[:, 50], label=f"Neuron {50}")
    plt.axhline(INPUT_AMP, color="r", ls="--", alpha=0.4)
    plt.xlabel("Time (s)")
    plt.ylabel("Current (nA)")
    plt.title(f"Trapezoid + 2hz Sinusoid input")

    os.makedirs("figures", exist_ok=True)
    if CCOV > 0 or ICOV > 0:
        plt.savefig(f"figures/CI_withNoise.png")
    else:
        plt.savefig(f"figures/CI.png")

    print("Figure saved in figures/.. folder")


def test_CoV():

    CI_DC = cortical_input(
        MN_POOL_SIZE,
        N_CLUST,
        INPUT_AMP,
        T_SIM,
        DT_SIM,
        CCOV,
        ICOV,
        "DC",
    )

    mean_activity = CI_DC.mean(axis=0)  # Mean cortical input per neuron
    std_activity = CI_DC.std(axis=0)  # Standard deviation per neuron

    CoV_CI_DC = std_activity / mean_activity
    print(f"max CoV for DC CI = {np.max(CoV_CI_DC)}")

    ## Results:
    # For CCov = 10 and ICoV = 5 -> output CoV = ca. 0.113...
    # For CCov = 20 and ICoV = 5 -> output CoV = ca. 0.207...
    # For CCov = 15 and ICoV = 5 -> output CoV = ca. 0.16...
    # For CCov = 0 and ICoV = 5 -> output CoV = ca. 0.051...
    # !!! For CCov = 14 and ICoV = 5 -> output CoV = ca. 0.15...


def _main():

    # plot_CI()
    test_CoV()

    # plt.figure(2, figsize=(12, 6))
    # sns.heatmap(
    #     CI.T, cmap="coolwarm", xticklabels=1000, yticklabels=5
    # )  # .T to transpose
    # plt.xlabel("Time (samples)")
    # plt.ylabel("Neurons")
    # plt.title("Cortical Input Heatmap")

    plt.show()


if __name__ == "__main__":
    _main()
