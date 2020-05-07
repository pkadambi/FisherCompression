% clear all
results_dir = './results/grid_x_-.75:.75:30_y-.75:.75:30_test';
model_names = {'distillation_teq1', 'distillation_teq2', 'distillation_teq3', 'distillation_teq4', 'STE', 'msqe', 'fisher'};
labels = {'Distil T=1', 'Distil T=2', 'Distil T=3', 'Distil T=4', 'STE', 'msqe', 'fisher'};

n_runs=10;
det_vals = parse_results(results_dir, n_runs, model_names);
dets = mean(det_vals);
accuracies = [93.39, 93.78, 94.05, 94.1, 93.15, 93.34,  93.31];

figure()
scatter(dets, accuracies, 'filled')
hold on
% accuracies = [93.15, 93.31, 93.39, 93.78, 94.05, 94.1];
text(dets-.02, accuracies-.02, labels)
xlabel(sprintf('Determinant of Hessian of 2D Fit Curve\n (Larger values imply greater curvature)'))
ylabel('Accuracy')
title(sprintf('Accuracy vs Determinant of Hessian of \n 2D Parabolic Curve Fit to Loss Landscape'))