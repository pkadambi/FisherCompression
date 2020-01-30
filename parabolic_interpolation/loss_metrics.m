clear all
%% Step 1: Import all data files
fisher_vals = importdata('fisher.txt');
dist_teq1 = importdata('distillation_teq1.txt');
dist_teq2 = importdata('distillation_teq2.txt');
dist_teq3 = importdata('distillation_teq3.txt');
dist_teq4 = importdata('distillation_teq4.txt');
ste_vals = importdata('STE.txt');
X = importdata('X.txt');
Y = importdata('Y.txt');
%TODO: add FP here, add MSQE here, possibly different quantization methods
%as well

labels = {'Fisher', 'Distil T=1', 'Distil T=2', 'Distil T=3', 'Distil T=4', 'STE'};
data = {fisher_vals, dist_teq1, dist_teq2, dist_teq3, dist_teq4, ste_vals};
accuracies = [93.31, 93.39, 93.78, 94.05, 94.1, 93.15];
hessians = cell(size(data));
%% Step 2: Evaluate metric of minimum for each point (fit 2nd order poly curve, get determiniant of hessian of the curve)
% contourf(X, Y, fisher_vals)
THRESHOLD = 3;
n_runs = size(data);

syms g(a,b)
syms a, b
dets= []
for ii = 1:n_runs(2)
    Z = data{ii};
    Z = Z(:);
    f = fit([X(:), Y(:)], Z, 'poly22', 'Exclude', Z>THRESHOLD );
    g(a,b) = f.p00 + f.p10*a+ f.p01*b+f.p20*a^2+f.p11 * a * b + f.p02 * b^2;
    % loss is thresholded to 3, since even @ the start of training, the
    % loss is less than 3
    Z(Z>THRESHOLD )=THRESHOLD+1e-8;
    
    hessians{ii} = vpa(hessian(g, [a,b]),7);
    disp(labels(ii))
    detval = double(det(hessians{ii}));
    dets = [dets detval];
    disp(det(hessians{ii}))
%     figure()
%     plot(f, [X(:), Y(:)],  Z )
%     title(strcat(labels{ii}, strcat(', Truncating loss less than 3, Hess Det: ', num2str(double(det(hessians{ii}))))))

end


%% Step 3: loop through cases and plot 
figure()
scatter(dets, accuracies, 'filled')
hold on
text(dets-.02, accuracies-.02, labels)
% for ii = 1:n_runs
%     
%     text(dets(ii)-.02, accuracies(ii)-.02, labels{ii}, 'FontSize', 8)
% 
% end
xlabel('Determinant of Hessian 2D Fit Curve')
ylabel('Accuracy')


