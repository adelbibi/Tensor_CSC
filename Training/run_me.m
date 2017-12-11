clear
close all
clc

global pars;
pars.trsize=[100 100];
pars.tssize=[100 100];
pars.verbose = 'csc';

pars.csc.CONTRAST_NORMALIZE = 'local_cn';
pars.csc.ZERO_MEAN = 1;
pars.csc.COLOR_IMAGES = 'gray';


%% Start parpool
poolobj = gcp('nocreate');
if(isempty(poolobj))
    parpool(12);
end

%% Select data
syntehtic_data = 0;
dataset_number = 3;
datasets ={...
    'City Dataset (10 images)',     '..\datasets\images\city_100_100',	'city';
    'Fruit Dataset (10 images)',	'..\datasets\images\fruit_100_100',   'fruit'
    'Pepper Dataset (1 images)',    '..\datasets\images\peppers',         'peppers'};
[D,Dhat,X,Xhat,Y,Yhat,params_sizes] = load_data(datasets,dataset_number,syntehtic_data);
n1 = params_sizes(1); n2 = params_sizes(2); n3 = params_sizes(3); n4 = params_sizes(4); N = params_sizes(5); K = params_sizes(6);
filter_szx = 5;
filter_szy = 5;
%% Problem parameters
lambda = 1e-1;
%% Optimization Parameters
error_obj = [];
counter = 0;
max_iter = 100;
error_obj_change_thresh = 1e-8;
error_reg_total_iter = [];
%% Fixed point optimization
while(true)
    counter = counter + 1;
    if(counter > max_iter)
        break;
    end
    %Fixing the dictionary and updating the sparse codes
    fprintf('-----> Updateing X (Sparse Code) \n');
    [X,error_XZnorm,error_reg_X] = sparse_code_update_ADMM_2D(Dhat,Xhat,Yhat,n3,n4,K,N,lambda);
    
    loss1 = lambda*sum(abs(X(:)));
    X_per = permute(X,[3, 4, 1, 2]);
    Xhat_per = fft2(X_per);
    Xhat = permute(Xhat_per,[3, 4, 1, 2]);
    
    %Fixing the sparse code and updating the dictionary
    fprintf('-----> Updateing D (Dictionary Learning) \n');
    [D,error_DTnorm,error_reg_D] = dictionary_update_ADMM_2D(Dhat,Xhat,Yhat,n1,n3,n4,K,N,filter_szx,filter_szy);
    
    loss2 = error_reg_D(end);
    D_per = permute(D,[3, 4, 1, 2]);
    Dhat_per = fft2(D_per);
    Dhat = permute(Dhat_per,[3, 4, 1, 2]);
    
    error_obj(counter) = loss1 + loss2;
     if (counter == 1)
        error_obj_change = 0 ;
    else
        error_obj_change = norm(error_obj(end) - error_obj(end-1))/norm(error_obj(end-1));
    end
    fprintf('----------------------------------------------------------------------------- \n');
    fprintf('+ Iter: %1.0f  RegError: %1.6f \n',counter,error_obj(counter));
    fprintf('----------------------------------------------------------------------------- \n');
    
    if(counter > 2)
        if(error_obj_change < error_obj_change_thresh)
            break;
        end
    end
end
save(strcat('results/dataset_',datasets{dataset_number,3},sprintf('_lambda=%0.5g_filter_X=%1.0f_filter_Y=%1.0f_NumberImages=%1.0f_NumberFilters=%1.0f.mat',lambda,filter_szx,filter_szy,N,K)),'D');
result_filename = strcat('dataset_', datasets{dataset_number,3}, sprintf('_lambda=%0.5g_filter_X=%1.0f_filter_Y=%1.0f_NumberImages=%1.0f_NumberFilters=%1.0f.txt',lambda,filter_szx,filter_szy,N,K));
fout = fopen(['results/' result_filename],'w');
fprintf(fout,'----------------------------------------------------------------------------------\n');
fprintf(fout,[' Dataset .......................... ', datasets{dataset_number,3} '\n']);
fprintf(fout,' Lambda ......................... %0.g\n', lambda);
fprintf(fout,' filter_size_X ......................... %d\n', filter_szx);
fprintf(fout,' filter_size_Y ......................... %d\n', filter_szy);
fprintf(fout,' Number of filters ......................... %d\n', K);
fprintf(fout,' Number of training examples ......................... %d\n', N);
fprintf(fout,' Max ADMM Iter ......................... %d\n', counter);
fprintf(fout,' loss1 ............. %0.g\n', loss1);
fprintf(fout,' loss2 ............ %0.g\n', loss2);
fprintf(fout,' error_obj ................... %0.g\n', error_obj(end));
fprintf(fout,' error_obj change ................... %.2g\n', error_obj_change);
fprintf(fout,'----------------------------------------------------------------------------------\n');
fclose(fout);
% save(strcat('..\output\dictonary_',datasets{dataset_number,3},'_Dictionary.mat'),'D')
% delete(poolobj);