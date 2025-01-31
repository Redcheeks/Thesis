import numpy as np


def trapez_current(duration, dt, max_current, ramp_time, hold_time):
    """
    Generates a current vector that ramps up linearly, holds, and then ramps down.

    Args:
        duration (float): Total duration of the current injection (ms).
        dt (float): Time step (ms).
        max_current (float): Peak current level (nA).
        ramp_time (float): Time taken to reach the max current (ms).
        hold_time (float): Time to hold the max current before falling (ms).

    Returns:
        np.array: Current vector.
    """
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
    spike_train = np.zeros(len(time) - 2 * len(ramp_up))

    for i in range(0, len(spike_train), spike_period):
        spike_train[i : i + int(0.1 * spike_period)] = (
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
