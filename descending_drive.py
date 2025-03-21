import numpy as np
import scipy.signal as signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns  # you may need to run:     conda install seaborn -c conda-forge and maybe update your environment

# to do: implement descending drives. look at robins code


def cortical_input(
    n_mn, n_clust, max_I, T_dur, dt, CCoV=20, ICoV=5, mean_shape="trapezoid", freq=2
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

    time = np.arange(0, T_dur, dt)

    if mean_shape == "trapezoid":
        # Compute the number of samples for each phase
        zero_len = int(round(0.00 * T_dur) / dt)  # Zero start and end
        ramp_len = int(round(0.16 * T_dur) / dt)  # Ramp-up and ramp-down
        hold_len = int(round(0.68 * T_dur) / dt)  # + 1  # Hold phase

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
        ramp_len = int(round(0.5 * T_dur) / dt)  # Ramp-up and ramp-down

        # Create each section
        ramp_up = np.linspace(0, max_I, ramp_len)
        ramp_down = np.flip(np.linspace(0, max_I, ramp_len))

        mean_drive = np.concatenate([ramp_up, ramp_down])

    elif mean_shape == "sinusoid.hz":

        time_sin = np.arange(0, T_dur, dt)  # Time vector

        # Generate the sinusoidal wave
        sine_wave = np.sin(-0.5 * np.pi + 2 * np.pi * freq * time_sin)

        # Normalize to range [0, max_I]
        mean_drive = max_I * (sine_wave + 1) / 2

    elif mean_shape == "step-sinusoid":

        # Compute the number of samples for each phase
        ramp_len = int(round((2 / 12) * T_dur) / dt)  # Ramp-up
        hold_len = int(round((4 / 12) * T_dur) / dt)  # 2 Hold phases

        # Generate the sinusoidal wave
        sine_wave = np.sin(-0.5 * np.pi + 2 * np.pi * freq * hold_len)

        # Normalize to range [0, max_I/2]
        sine_drive = max_I / 2 * (sine_wave + 1) / 2

        # Create each section
        ramp_up = np.linspace(0, 0.5 * max_I, ramp_len)
        hold_segment = np.full(hold_len, 0.5 * max_I)
        sine_segment = (
            hold_segment + sine_drive
        )  # here we will have sinusoid added to hold
        ramp_down = np.linspace(0, 0.5 * max_I, ramp_len)

        mean_drive = np.concatenate([ramp_up, hold_segment, sine_segment, ramp_down])

    else:  # elif mean_shape == "step":

        # Compute the number of samples for each phase
        ramp_len = int(round((2 / 12) * T_dur) / dt)  # Ramp-up
        hold_len = int(round((4 / 12) * T_dur) / dt)  # Hold phase

        # Create each section

        ramp_up_1 = np.linspace(0, 0.5 * max_I, ramp_len)
        hold_segment_1 = np.full(hold_len, 0.5 * max_I)
        ramp_up_2 = np.linspace(0.5 * max_I, max_I, ramp_len)
        hold_segment_2 = np.full(hold_len, max_I)

        mean_drive = np.concatenate(
            [ramp_up_1, hold_segment_1, ramp_up_2, hold_segment_2]
        )

    mean_CI = max(mean_drive)

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


def _main():
    # Example Usage
    T_dur = 1000  # Total time in ms
    dt = 1  # Time step in ms
    n_mn = 300  # Number of motor neurons
    n_clust = 5  # Number of clusters
    max_I = 10  # Max input current (nA)
    CCoV = 20  # Common noise CoV (%)
    ICoV = 1  # Independent noise CoV (%)

    CI = cortical_input(n_mn, n_clust, max_I, T_dur, dt, CCoV, ICoV, "trapezoid")

    # Plot the first motor neuron's cortical input
    plt.figure(1, figsize=(8, 6))
    time = np.linspace(0, T_dur, CI.shape[0])
    for i in range(5):  # Plot first 5 neurons
        plt.plot(time, CI[:, i], label=f"Neuron {i+1}")

    plt.plot(time, CI[:, 100], label=f"Neuron {100}")
    plt.plot(time, CI[:, 200], label=f"Neuron {200}")
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (nA)")
    plt.title("Cortical Input for Multiple Neurons")

    plt.figure(2, figsize=(12, 6))
    sns.heatmap(
        CI.T, cmap="coolwarm", xticklabels=1000, yticklabels=5
    )  # .T to transpose
    plt.xlabel("Time (samples)")
    plt.ylabel("Neurons")
    plt.title("Cortical Input Heatmap")
    plt.show()

    mean_activity = CI.mean(axis=0)  # Mean cortical input per neuron
    std_activity = CI.std(axis=0)  # Standard deviation per neuron

    print(std_activity / mean_activity)

    # print("Mean cortical input per neuron:")S
    # print(mean_activity)

    # print("\nStandard deviation per neuron:")
    # print(std_activity)


if __name__ == "__main__":
    _main()
