import numpy as np
import matplotlib.pyplot as plt


def trapez_current(duration, dt, max_current, ramp_time):
    """
    Generates a current vector that ramps up linearly, holds, and then ramps down.

    Args:
        duration (float): Total duration of the current injection (ms).
        dt (float): Time step (ms).
        max_current (float): Peak current level (nA).
        ramp_time (float): Time taken to reach the max current (ms).


    Returns:
        np.array: Current vector.
    """
    hold_time = duration - 2 * ramp_time
    time = np.arange(0, duration, dt)

    # Ramp-up phase
    ramp_up = np.linspace(0, max_current, int(ramp_time / dt))

    # Hold phase
    hold = np.full(int(hold_time / dt), max_current)

    # Ramp-down phase
    ramp_down = np.linspace(max_current, 0, int(ramp_time / dt))

    # Fill the remaining time with zeros
    remaining_time = max(0, len(time) - (len(ramp_up) + len(hold) + len(ramp_down)))
    tail = np.zeros(remaining_time)

    # Concatenate all phases
    current = np.concatenate([ramp_up, hold, ramp_down, tail])

    return time, current


def linear_spiking_current(
    duration, dt, max_current, ramp_time, spike_amplitude, spike_frequency
):
    """
    Generates a current vector that ramps up linearly, then spikes at 2Hz frequency, and then ramps down.

    Args:
        duration (float): Total duration of the current injection (ms).
        dt (float): Time step (ms).
        max_current (float): Peak current level (nA).
        ramp_time (float): Time taken to reach the max current (ms).
        spike_amplitude (float): Amplitude of the spikes above max current (nA).
        spike_frequency (float): Frequency of spiking (Hz).

    Returns:
        np.array: Current vector.
    """
    time = np.arange(0, duration, dt)

    # Ramp-up phase
    ramp_up = np.linspace(0, max_current, int(ramp_time / dt))

    # Spiking phase
    spike_period = int((1 / spike_frequency) * 1000 / dt)  # Convert Hz to ms intervals
    spike_train = np.ones(len(time) - 2 * len(ramp_up)) * max_current

    for i in range(0, len(spike_train), spike_period):
        spike_train[i : i + int(0.5 * spike_period)] = (
            spike_amplitude + max_current
        )  # Short pulse

    # Ramp-down phase
    ramp_down = np.linspace(max_current, 0, int(ramp_time / dt))

    # Ensure lengths match
    remaining_time = max(
        0, len(time) - (len(ramp_up) + len(spike_train) + len(ramp_down))
    )
    tail = np.zeros(remaining_time)

    # Concatenate all phases
    current = np.concatenate([ramp_up, spike_train, ramp_down, tail])

    return time, current


def _main():
    # Example Usage
    duration = 1000  # Total time in ms
    dt = 1  # Time step in ms
    max_current = 200  # Peak current in pA
    ramp_time = 200  # Time to reach peak in ms
    hold_time = 300  # Time to hold peak in ms
    spike_amplitude = 100  # Additional amplitude of spikes in pA
    spike_frequency = 20  # Spikes at 2 Hz

    # Generate current traces
    time1, current1 = trapez_current(duration, dt, max_current, ramp_time)
    time2, current2 = linear_spiking_current(
        duration, dt, max_current, ramp_time, spike_amplitude, spike_frequency
    )

    # Plot results
    plt.figure(figsize=(12, 5))

    plt.subplot(2, 1, 1)
    plt.plot(time1, current1, label="Linear Hold & Fall Current")
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (pA)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time2, current2, label="Linear Spiking Current (2 Hz)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Current (pA)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _main()
