%% Quick Sorting Algorithm for Neuron Discharge Times in MATLAB
% Based on 5 firings in 1 second = Active
file_name = 'trapezoid5mvc_repetitive_doublets';
load(file_name)


% Initialize activation times for each neuron
activation_times = inf(1, length(discharge_times));

% Define ignore period (e.g., first 5 seconds)
ignore_period = 10 * fs; % Convert to samples


% Calculate activation times based on the 8th firing within 1 second
min_fires = 10; %amount of spikes in a timeframe
per_time = 1.5;% time frame

for i = 1:length(discharge_times)
    times = discharge_times{i} / fs;
    for j = 1:length(times) - min_fires 
        if times(j + min_fires) - times(j) <= per_time && times(j+min_fires) >= ignore_period / fs
            activation_times(i) = times(j);
            break;
        end
    end
end

% Sort the neurons by their activation times in ascending order
[~, sorted_indices] = sort(activation_times);
discharge_times = discharge_times(sorted_indices);
source_signals = source_signals(sorted_indices);
% Display the sorted discharge times
disp('Sorted Discharge Times by Activation:');
disp(discharge_times);

% Save the sorted variables to a new .mat file
save(file_name + "_sorted.mat", 'discharge_times', 'force', 'fs', 'source_signals');

%% Quick Sorting Algorithm for Neuron Discharge Times in MATLAB (Weighted Activation)
file_name = 'trapezoid5mvc_repetitive_doublets';
load(file_name)

% Initialize activation times for each neuron
activation_times = inf(1, length(discharge_times));

% Define ignore period (e.g., first 5 seconds)
ignore_period = 10 * fs; % Convert to samples

% Calculate weighted activation times based on 5 firings within 1 second
for i = 1:length(discharge_times)
    times = discharge_times{i} / fs;  % Convert times to seconds
    weights = [];  % To store activation times

    for j = 1:length(times) - 9
        if times(j + 9) - times(j) <= 2 && times(j) >= ignore_period / fs
            weights(end + 1) = times(j + 9); % Store activation time
        end
    end

    if ~isempty(weights)
        % Calculate weighted activation time (later activations matter more)
        weighted_time = sum(weights .* (1:length(weights))) / sum(1:length(weights));
        activation_times(i) = weighted_time;
    end
end

% Sort the neurons by their weighted activation times in ascending order
[~, sorted_indices] = sort(activation_times);
discharge_times = discharge_times(sorted_indices);
source_signals = source_signals(sorted_indices);

% Display the sorted discharge times
disp('Sorted Discharge Times by Weighted Activation (in seconds):');
disp(discharge_times);

% Save the sorted variables to a new .mat file
save(file_name + "_weighted_sorted.mat", 'discharge_times', 'force', 'fs', 'source_signals');

%% Quick Sorting Algorithm for Neuron Discharge Times in MATLAB (Weighted Sustained Activation)
file_name = 'trapezoid5mvc_repetitive_doublets';
load(file_name)

% Initialize activation times for each neuron
activation_times = inf(1, length(discharge_times));

% Define ignore period (e.g., first 5 seconds)
ignore_period = 10 * fs; % Convert to samples

% Calculate weighted sustained activation times based on 10 firings within 2 seconds
for i = 1:length(discharge_times)
    times = discharge_times{i} / fs;  % Convert times to seconds
    sustained_activations = [];  % To store all sustained activation times

    for j = 1:length(times) - 9
        if times(j + 9) - times(j) <= 2 && times(j) >= ignore_period / fs
            sustained_activations(end + 1) = times(j + 9); % Store each sustained activation time
        end
    end

    if ~isempty(sustained_activations)
        % Calculate weighted average of sustained activation times (later activations weighted more)
        weights = 1:length(sustained_activations);
        weighted_time = sum(sustained_activations .* weights) / sum(weights);
        activation_times(i) = weighted_time;
    end
end

% Sort the neurons by their weighted sustained activation times in ascending order
[~, sorted_indices] = sort(activation_times);
discharge_times = discharge_times(sorted_indices);
source_signals = source_signals(sorted_indices);

% Display the sorted discharge times
disp('Sorted Discharge Times by Weighted Sustained Activation (in seconds):');
disp(discharge_times);

% Save the sorted variables to a new .mat file
save(file_name + "_sustained_weighted_sorted.mat", 'discharge_times', 'force', 'fs', 'source_signals');
