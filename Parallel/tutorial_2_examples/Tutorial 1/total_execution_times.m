cd '..\x64\Debug\Tutorial 1'

results = []
labels = []

for i = 1:6
    label = strcat("cpu_total_", string(10^i))
    filename = strcat(label, ".csv")    
    results = [results; readmatrix(filename)]
    labels = [labels; repmat(label, 10, 1)]
    
    label = strcat("gpu_total_", string(10^i))
    filename = strcat(label, ".csv")    
    results = [results; readmatrix(filename)]
    labels = [labels; repmat(label, 10, 1)]
end

figure
boxplot(results, labels, 'Colors', 'rb')
ylabel("Total execution time [us]")