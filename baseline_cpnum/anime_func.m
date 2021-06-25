function [AUCs] = anime_func(runs,dim)
addpath('../tensor_toolbox-master');
addpath('../poblano_toolbox-master');
rng('default');


nvec = [25838,4066];
nmod = length(nvec);
dim = dim;

runs = runs;
total = zeros(runs,1);

AUCs = zeros(length(runs),1);
MAE = zeros(length(runs),1); 
times = zeros(length(runs),1); 
for i = 1:runs
    train_path = strcat( '..\..\..\..\Dropbox\Tensor dataset\anime\anime_train_', num2str(i), '.txt' );
    test_path = strcat( '..\..\..\..\Dropbox\Tensor dataset\anime\anime_test_', num2str(i), '.txt' );
    
    train = load(train_path);
    test = load(test_path);
    
    train(:,1:nmod) = train(:,1:nmod)+1;
    test(:,1:nmod) = test(:,1:nmod)+1;
    
    trind = (train(:,1:nmod));
    trvals = train(:,nmod+1);
    
    Y = sptensor(trind, trvals,nvec);
    
    
    %P = cp_wopt(Y,W,dim, 'init', 'random','alg', 'lbfgs');
    
    start_time = clock;
    
    P = cp_nmu(Y,dim);
    
    end_time = clock;
    %for t=1:length(test)
        %init with randn
        %P = cp_wopt(Y,W,dim,'alg', 'lbfgs');
        %prediction
        sub = test(:,1:nmod);
        ymiss = test(:,nmod+1);

        nmod = length(nvec);
        pred = ones(length(ymiss),dim);
        for j=1:nmod
            pred = pred .* P.u{j}(sub(:,j),:);
        end
        pred = pred * fliplr(P.lambda')';
        [~,~,~,auc] = perfcurve(ymiss,pred,1);
       AUCs(i) = auc;
      
       times(i) = etime(end_time,start_time);
        %fprintf('mse = %g\n',val);
    %end
    fprintf('run %d,  AUC = %g \n', i, auc);
    
    %total(i) = mean(mses);
end

 
fileID = fopen('./result_log/anime_nmu.txt','a+');
fprintf(fileID,'runs = %d, rank = %d \n',runs ,dim);
fprintf(fileID,'mean AUC = %g, std AUC = %g \n', mean(AUCs),std(AUCs));
fprintf(fileID,'training time is %g \n',  mean(times));
fprintf(fileID,'\n \n \n');
fclose(fileID);
%fprintf('average = %g, std = %g', mean(total), std(total)/sqrt(runs));
