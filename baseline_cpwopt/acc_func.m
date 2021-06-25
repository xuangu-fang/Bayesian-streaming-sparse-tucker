function [rmses] = acc_func(dim)
addpath('../tensor_toolbox_2.5');
addpath('../poblano_toolbox-master');
rng('default');


nvec = [3000,150,30000];
nmod = length(nvec);
dim = dim;

runs = 1;
total = zeros(runs,1);

rmses = zeros(length(runs),1);
MAE = zeros(length(runs),1); 
times = zeros(length(runs),1); 
for run = 1:runs
    

train_path = '..\..\..\..\Dropbox\Tensor dataset\dm_data\acc\ibm-large-tensor.txt';
test_path ='..\..\..\..\Dropbox\Tensor dataset\dm_data\acc\ibm.mat';;
    
    train = load(train_path);
    test = load(test_path);
    test = test.data.test;
    
    trind = (train(:,1:nmod))+1;
    trvals = train(:,nmod+1);
    
    Y = sptensor(trind, trvals,nvec);
    W = sptensor(trind, ones(length(trind),1), nvec);
    
    
    
    start_time = clock;
    
    %P = cp_nmu(Y,dim);
    %P = cp_als(Y,dim, 'init', 'random');
    P = cp_wopt(Y,W,dim, 'init', 'random');
    end_time = clock;
    
    rmses = zeros(50,1);
    
    for i=1:50
    %for t=1:length(test)
        %init with randn
        %P = cp_wopt(Y,W,dim,'alg', 'lbfgs');
        %prediction
        sub = test{i}.subs;
        ymiss = test{i}.Ymiss;

        nmod = length(nvec);
        pred = ones(length(ymiss),dim);
        for j=1:nmod
            pred = pred .* P.u{j}(sub(:,j),:);
        end
        pred = pred * fliplr(P.lambda')';
        val = sqrt(mean((ymiss- pred).^2));
       rmses(i) = val;
       MAE(i) = mean(abs(ymiss- pred));
       times(i) = etime(end_time,start_time);
        
    end
        %fprintf('mse = %g\n',val);
    %end
%     fprintf('run %d,  rmse = %g \n', i, val);
%     fprintf(' MAE= %g \n',  mean(MAE));
    %total(i) = mean(mses);
end
fprintf(' mean rmse= %g \n',  mean(rmses));
fileID = fopen('./result_log/acc_cpwopt.txt','a+');
fprintf(fileID,'runs = %d, rank = %d \n',runs ,dim);
fprintf(fileID,'mean rmse = %g, std rmse = %g \n', mean(rmses),std(rmses));
fprintf(fileID,'mean MAE= %g, std MA = %g \n',  mean(MAE),std(MAE));
fprintf(fileID,'training time is %g \n',  mean(times));
fprintf(fileID,'\n \n \n');
fclose(fileID);
%fprintf('average = %g, std = %g', mean(total), std(total)/sqrt(runs));
