cd '..\x64\Debug\Tutorial 1'

runs = 10;

results = [];
labels = [];

for i = 1:6
    prefix = strcat("cpu_", string(10^i));
    filename = strcat(prefix, ".csv");
    data = readmatrix(filename);
    results = [results; data(:)];
    labels = [labels; repmat(strcat(prefix, "_a"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_b"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_c"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_kernel"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_total"), runs, 1)];
end

for i = 1:6
    prefix = strcat("gpu_", string(10^i));
    filename = strcat(prefix, ".csv");
    data = readmatrix(filename);
    results = [results; data(:)];
    labels = [labels; repmat(strcat(prefix, "_a"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_b"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_c"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_kernel"), runs, 1)];
    labels = [labels; repmat(strcat(prefix, "_total"), runs, 1)];
end

figure;
boxplot(results, labels, 'Colors', 'rgbmk');
ylabel("Time [ns]");
title('Memory transfer, kernel execution, and total execution times ');

results = [];
labels = [];

for i = 1:6
    prefix = strcat("cpu_", string(10^i));
    filename = strcat(prefix, ".csv");
    data = readmatrix(filename);
    results = [results; data(:, 4)];
    labels = [labels; repmat(strcat(prefix, "_kernel"), runs, 1)];
    
    prefix = strcat("gpu_", string(10^i));
    filename = strcat(prefix, ".csv");
    data = readmatrix(filename);
    results = [results; data(:, 4)];
    labels = [labels; repmat(strcat(prefix, "_kernel"), runs, 1)];
end

figure
boxplot(results, labels, 'Colors', 'rb');
ylabel("Time [ns]");
title('Kernel execution times ');

results = [];
labels = [];

for i = 1:6
    prefix = strcat("cpu_", string(10^i));
    filename = strcat(prefix, ".csv");
    data = readmatrix(filename);
    results = [results; data(:, 5)];
    labels = [labels; repmat(strcat(prefix, "_total"), runs, 1)];
    
    prefix = strcat("gpu_", string(10^i));
    filename = strcat(prefix, ".csv");
    data = readmatrix(filename);
    results = [results; data(:, 5)];
    labels = [labels; repmat(strcat(prefix, "_total"), runs, 1)];
end

figure;
boxplot(results, labels, 'Colors', 'rb');
ylabel("Time [ns]");
title('Total execution times ');
