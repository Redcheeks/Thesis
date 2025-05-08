import scipy.io
import numpy as np
import matplotlib.pyplot as plt


def _main():

    trapezoid_sinusoid2hz = scipy.io.loadmat(
        "Experimental Data Analysis/trapezoid10mvc_sinusoid2hz5to15mvc.mat"
    )

    trapezoid = scipy.io.loadmat("Experimental Data Analysis/trapezoid20mvc.mat")

    data_to_plot = trapezoid
    plt.ion()  # Turn on interactive mode

    for ind in np.arange(
        data_to_plot["discharge_times"][0].shape[0]
    ):  # for each neuron
        plt.close()
        fig, ax = plt.subplots()
        # plot , frequency.
        ax.plot(
            data_to_plot["discharge_times"][0][ind][0][:-1] / 2e3,
            1000 / (np.diff(data_to_plot["discharge_times"][0][ind][0][:]) / 2),
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
