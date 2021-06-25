function [rmses] = alog_func(runs,dim)
addpath('../tensor_toolbox-master');
addpath('../poblano_toolbox-master');
rng('default');


nvec = [200,100,200];

dim = dim;

runs = runs;
total = zeros(runs,1);

rmses = zeros(length(runs),1);
MAE = zeros(length(runs),1); 
times = zeros(length(runs),1); 
for i = 1:runs
    
train_path = strcat( '..\..\..\..\Dropbox\Tensor dataset\alog_data\train-fold-', num2str(i), '.txt' );
test_path = strcat( '..\..\..\..\Dropbox\Tensor dataset\alog_data\test-fold-', num2str(i), '.txt' );
    
    train = load(train_path);
    test = load(test_path);
    
    trind = (train(:,1:3));
    trvals = train(:,4);
    
    Y = sptensor(trind, trvals,nvec);
    
    
    %P = cp_wopt(Y,W,dim, 'init', 'random','alg', 'lbfgs');
    
    start_time = clock;
    
    P = cp_nmu(Y,dim);
    
    end_time = clock;
    %for t=1:length(test)
        %init with randn
        %P = cp_wopt(Y,W,dim,'alg', 'lbfgs');
        %prediction
        sub = test(:,1:3);
        ymiss = test(:,4);

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
        %fprintf('mse = %g\n',val);
    %end
    fprintf('run %d,  rmse = %g \n', i, val);
    fprintf(' MAE= %g \n',  mean(MAE));
    %total(i) = mean(mses);
end

 
fileID = fopen('./result_log/alog_nmu.txt','a+');
fprintf(fileID,'runs = %d, rank = %d \n',runs ,dim);
fprintf(fileID,'mean rmse = %g, std rmse = %g \n', mean(rmses),std(rmses));
fprintf(fileID,'mean MAE= %g, std MA = %g \n',  mean(MAE),std(MAE));
fprintf(fileID,'training time is %g \n',  mean(times));
fprintf(fileID,'\n \n \n');
fclose(fileID);
%fprintf('average = %g, std = %g', mean(total), std(total)/sqrt(runs));
