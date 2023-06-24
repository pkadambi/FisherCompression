clear
metric = 'trace';
results_dir = './results/grid_x_-.75:.75:30_y-.75:.75:30_test';
model_names = {'distillation_teq1', 'distillation_teq2', 'distillation_teq3', 'distillation_teq4', 'STE', 'msqe', 'fisher'};
labels = {'Distil T=1', 'Distil T=2', 'Distil T=3', 'Distil T=4', 'STE', 'MSQE', 'Fisher'};

n_runs=10;
metric_vals = parse_results(results_dir, n_runs, model_names, metric);
metrics= mean(metric_vals);
accuracies = [93.39, 93.78, 94.05, 94.1, 93.15, 93.34,  93.31];



% % det_vals(4:6,:)=[]
std_errors = std(metric_vals);

figure()
scatter(accuracies, metrics,  'filled')
hold on
errorbar(accuracies, metrics, std_errors, 'LineStyle', 'none')

% scatter(dets, accuracies, 'filled')
% errorbar(dets, accuracies, std_errors, 'LineStyle', 'none')
hold on
% accuracies = [93.15, 93.31, 93.39, 93.78, 94.05, 94.1];

offsets  = [ -.04 .1; -.02 .1; .23 1; .1 1.5; -.01 -2; .13 2; -.04 2];

for ii=1:length(accuracies)
    text(accuracies(ii)-offsets(ii,1), metrics(ii)-offsets(ii,2), labels{ii}, 'FontSize', 12)
end

% text(accuracies-.02, dets-13, labels, 'FontSize', 12)
% xlabel(sprintf('Determinant of Hessian of 2D Fit Curve\n (Larger values imply greater curvature)'))
% ylabel('Accuracy')
% title(sprintf('Accuracy vs Determinant of Hessian of \n 2D Parabolic Curve Fit to Loss Landscape'))

xlabel(sprintf('Test Accuracy CIFAR 10'), 'fontweight', 'bold', 'fontsize', 15)
ylabel({'Trace of Hessian of Parabolic Fit'; ' (Loss Flatness)'}, 'fontweight', 'bold', 'fontsize', 15)
% title(sprintf(''))

grid on
xlim([93.1, 94.25])
disp()