function [rmses] = alog_func(runs,R,batch_size)

addpath_recurse('./tensor_toolbox-master');
addpath_recurse('./poblano_toolbox');
addpath_recurse('L1General');
addpath_recurse('./lightspeed');
addpath_recurse('./unity_funcs');

rng('default');
nvec = [200,100,200];
nmod = length(nvec);



%settings
batch_size = batch_size;
runs=runs;
R = R;
R_list= R*[1,1,1];
v_U = 1;
v_W = 1;
%para for exact prior
rho_0 = 0.3;
sigma_0 = 1;
T =5;
damping = 0.95;

opt = [];
opt.max_iter =1;
opt.tol = 1e-2;

%diary alog_small_test
% data = load(strcat(strcat('./small/alog_fold',num2str(1)),'.mat'));
% data = data.data;
% 
% 
% data.Y = tenfun(@log, tensor(data.Y) + 1);
% data.train = find(data.Y~=0);
% data.Ymiss = log(data.test_vals+1);
% 
% test_X =  data.test_ind;
% test_Y = log(data.test_vals+1);
% train = [data.train,data.Y(data.train)];

rmses = zeros(length(runs),1);
MAEs = zeros(length(runs),1); 

for run_num = 1:runs

train_path = strcat( '.\data\alog\train-fold-', num2str(run_num), '.txt' );
test_path = strcat( '.\data\alog\test-fold-', num2str(run_num), '.txt' );
    
train = load(train_path);
test = load(test_path);

test_X =   test(:,1:3);
test_Y =  test(:,4);

% shuffel
train = train(randperm(size(train,1)),:);

% init vec(W): core tensor
W_size = prod(R_list);
W_mean = rand(W_size,1);
W_var = v_W*eye(W_size);

% init W_sparse_prior:
W_prior_rho = 0.1*rho_0*ones(W_size,1);
W_prior_mean = W_mean;
W_prior_var = v_W*ones(W_size,1);

% init embedding U
U_mean = cell(nmod,1);
U_var = cell(nmod, 1);

for k=1:nmod
    U_mean{k} = rand(nvec(k),R_list(k));
    U_var{k} = reshape(repmat(v_U*eye(R_list(k)), [1, nvec(k)]), [R_list(k),R_list(k),nvec(k)]);
    % use diag var
    %U_var{k}= v_U*ones(nvec(k),R_list(k));
end

% embedding
prior_a = 0.001;
prior_b = 0.001;
U = {U_mean, U_var, [prior_a, prior_b],R_list};
W = {W_mean,W_var};
W_prior = {W_prior_rho,W_prior_mean,W_prior_var,rho_0,sigma_0};




data = train;
n = size(train,1);
iter=0;

prior_update = 0;

for i = 1:batch_size:n
        iter = iter+1;
        if i+batch_size-1 <= n
            batch_data = data(i:i+batch_size-1, :);
            batch_y = data(i:i+batch_size-1,end);
            batch_n = batch_size; % num of entries in current mini-batch
            
        else
            batch_data = data(i:end,:);
            batch_y = data(i:end,end);
            batch_n = size(batch_y,1);
        end
        
        if mod(iter,T)==0
            prior_update=1;
        end
        
        
       post = bayes_sparse_tucker_streaming_seq(U,W,W_prior,batch_data,opt,test_X,test_Y,prior_update,damping);
%      post = bayes_sparse_tucker_streaming_para(U,W,W_prior,batch_data,opt,test_X,test_Y,prior_update);
        prior_update=0;
        
        fprintf('iter = %d/%d finshed \n', iter,n/batch_size) ;

U_mean_post = post{1};
U_var_post = post{2};
W_vec_mean_post = post{3};
W_vec_var_post = post{4};
W_post_rho = post{5};
W_post_mean= post{6};
W_post_var= post{7};
ab_post = post{8};
a_post = ab_post(1);
b_post = ab_post(2);

U = {U_mean_post, U_var_post, [a_post, b_post],R_list};
W = {W_vec_mean_post,W_vec_var_post};
W_prior = {W_post_rho,W_post_mean,W_post_var,rho_0,sigma_0};


end % end of batch

% test part
test_len = size(test_X,1);

bi = U_mean_post{nmod}(test_X(:,nmod),:);
for j=nmod-1:-1:1
    mode=j;
    cur_term = U_mean_post{mode}(test_X(:,mode),:);
    bi = kron_prod_rows_vec(bi,cur_term);
end
pred_result  = bi*W_vec_mean_post; %size: n_test,1
rmses(run_num) = sqrt(sum((test_Y-pred_result).^2)/test_len);
MAEs(run_num) = mean(abs(test_Y-pred_result)); 
sparse_ratio = sum(W_post_rho<0)/W_size;
fprintf(' run %d,rmse = %g, sparse_ratio = %g\n',run_num, rmses(run_num),sparse_ratio);

end

 fprintf('mean rmse = %g, std rmse = %g \n', mean(rmses),std(rmses));
 fprintf('mean MAE= %g, std MA = %g \n',  mean(MAEs),std(MAEs));
 
fileID = fopen('./result_log/alog_st.txt','a+');
fprintf(fileID,'runs =%d, rank = %d, batch_size = %d \n',runs, R,batch_size);
fprintf(fileID,'mean rmse = %g, std rmse = %g, exact values are \n', mean(rmses),std(rmses));
fprintf(fileID,'mean MAE= %g, std MA = %g \n',  mean(MAEs),std(MAEs));
fprintf(fileID,'\n \n \n');
fclose(fileID);
end
