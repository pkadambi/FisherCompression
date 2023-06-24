function [det_vals] = parse_results(results_dir, n_runs, model_names, metric)

    data_dirs = strcat(results_dir,'/%s/%s.txt');

    X = importdata(strcat(results_dir, '/X.txt'));
    Y = importdata(strcat(results_dir, '/Y.txt'));

    n_models = size(model_names);
    n_models = n_models(2);

    j=0;

    det_values = zeros(n_runs, n_models);
    
    THRESHOLD = 3;

    while j<n_runs
    %     labels = { 'STE', 'Fisher', 'Distil T=1', 'Distil T=2', 'Distil T=3', 'Distil T=4',};
    %     curr_data = 

        for k=1:n_models

            datafile = sprintf(data_dirs, num2str(j), model_names{k});
            disp(datafile)
            loss_data = importdata(datafile);
            loss_data = loss_data(:);
            det_values(j+1,k)=poly_reg(X, Y, loss_data, THRESHOLD, metric);

        end

        j=j+1;
        
    end
    det_vals = det_values
end


% dirfile = './results/grid_x_-.75:.75:30_y-.75:.75:30_test/%s/%s'
% model_names = ''






