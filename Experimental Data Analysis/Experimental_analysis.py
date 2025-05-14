import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def _main():

    trapezoid_sinusoid2hz = scipy.io.loadmat(
        "Experimental Data Analysis/trapezoid10mvc_sinusoid2hz5to15mvc.mat"
    )

    trapezoid = scipy.io.loadmat("Experimental Data Analysis/trapezoid20mvc.mat")

    data_to_plot = trapezoid_sinusoid2hz
    plt.ion()  # Turn on interactive mode

    # Extract data
    discharge_times = data_to_plot["discharge_times"]
    fs = data_to_plot["fs"].item()  # Sampling frequency

    for ind, neuron_data in enumerate(discharge_times[0]):  # for each neuron
        plt.close()
        fig, ax = plt.subplots()
        spikes = np.sort(neuron_data.flatten())

        isi = np.diff(spikes)
        isi_freq = 1 / (isi / fs)
        isi_freq = np.insert(isi_freq, 0, 0)  # first spike frequency is 0

        ax.plot(
            spikes / fs,
            isi_freq,
            "o",
        )
        ax.set_ylabel("Instantaneous Frequency [Hz]")
        ax.set_xlabel("Time [s]")
        ax.set_title("Neuron number" + str(ind))
        plt.show()
        # plt.pause()
        input("Press Enter to plot the next point...")


if __name__ == "__main__":
    _main()
