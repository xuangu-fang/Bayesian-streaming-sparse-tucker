function [post] = bayes_sparse_tucker_streaming_seq(U,W,W_prior,train,opt,test_X,test_Y, prior_update,damping)

    damping_alpha = damping;

    U_mean = U{1};
    U_var = U{2};
    
    prior_a = U{3}(1);
    prior_b = U{3}(2);
    
    R_list = U{4};
    
    W_mean = W{1};
    W_var = W{2};
    
    W_prior_rho = W_prior{1};
    W_prior_mean = W_prior{2};
    W_prior_var = W_prior{3};
    rho_0 = W_prior{4};
    sigma_0 = W_prior{5};
    
    nmod = length(U_mean);
    R = size(U_mean{1}, 2);
    
    data = train;
    
    %indices of unique rows in each mode
    uind = cell(nmod, 1);
    %associated training entries
    data_ind = cell(nmod, 1);
    for k=1:nmod
        [uind{k}, ~, ic] = unique(data(:,k));
        data_ind{k} = cell(length(uind{k}),1);
        for j=1:length(uind{k})
            data_ind{k}{j} = find(ic == j);
        end
    end
    
    %init post usingf prior
    U_mean_post = U_mean;
    U_var_post = U_var;
    
    U_mean_post_old = U_mean;
    U_var_post_old = U_var;
    
    a_post = prior_a;
    b_post = prior_b;
    
    a_post_old = prior_a;
    b_post_old = prior_b;
    
    W_vec_mean_post = W_mean;
    W_vec_var_post = W_var;
    
    W_vec_mean_post_old = W_mean;
    
    W_post_rho = W_prior_rho;
    W_post_mean = W_prior_mean;
    W_post_var = W_prior_var;
    
    
    W_post_rho_old = W_prior_rho;
    W_post_mean_old = W_prior_mean;
    W_post_var_old = W_prior_var;
    
    %update
    y = data(:,end);
    tau = prior_a/prior_b;
    n = size(data,1);

    for iter = 1:opt.max_iter
        
        %old_u is used to compute the tol
        old_u = cell(nmod,1);
        for k=1:nmod
            if uind{k} ==0
                uind{k};
            end
            old_u{k} = U_mean_post{k}(uind{k},:);
        end
        
        
        for k=1:nmod
            %build some common terms
            %?? for j = 1: uniq-ind in mode k
            %fold core-tensor at k-mode 
            W_tensor = reshape(W_vec_mean_post,R_list); % define new tensor object here
            
            %shape: R_k*R_prod/R_k
            W_tensor_k = tenmat(W_tensor,k).data; 
            
            other_modes = setdiff(1:nmod,k);
            init_mod = other_modes(end);
            
            % kron_part of E(ai),init size: n*R_end, get the last mode as init
            kron_b_nk = U_mean_post{init_mod}(data(:,init_mod),:);
            
            % kron_part of E(aiai'),init size: R_end*R_end*n
            kron_b_nk2= U_var_post{init_mod}(:,:,data(:,init_mod))+ outer_prod_rows(kron_b_nk);
            
            for j=length(other_modes)-1:-1:1
                mod = other_modes(j);
                cur_b = U_mean_post{mod}(data(:,mod),:);
                kron_b_nk = kron_prod_rows_vec(kron_b_nk,cur_b);% (n,R_k)
                kron_b_nk2 = kron_prod_rows_mat(kron_b_nk2,U_var_post{mod}(:,:,data(:,mod))+ outer_prod_rows(cur_b));
            end
            % kron_b_nk final shape: n* R_prod/R_k
            % kron_b_nk2 final shape: R_prod/R_k * R_prod/R_k * n
            
            %after first ttm: size: R_k * R_prod/R_k * n  ttm R_k*R_prod/R_k
            
             %after second ttm: size: R_k * R_k * n
            
            E_ai = kron_b_nk*W_tensor_k'; %size(n,R_k)
            E_ai2 = ttm(ttm(tensor(kron_b_nk2),W_tensor_k,1),W_tensor_k,2);%size(R_k,R_k,n)
            
            
            % update U
            for j=1:length(uind{k})
                uid = uind{k}(j); % id of embedding
                eid = data_ind{k}{j};% id of associated entries
                
                a = E_ai(eid,:);
           
                a2 = E_ai2(:,:,eid);
                a2 = a2.data;
                
                inv_old_var = inv(U_var_post_old{k}(:, :, uid));
                %inv_old_var = inv(U_var_post{k}(:, :, uid));
                
                U_var_post{k}(:, :, uid) = inv(tau*sum(a2, 3)+ inv_old_var);
                U_mean_post{k}(uid,:)= U_var_post{k}(:, :, uid)*...
                    ( inv_old_var*reshape(U_mean_post{k}(uid,:),[R,1])+ reshape(a'*y(eid)*tau,[R,1]));
            end
            

        end % end of iter on mode
        
            % update vec(W)
            
            % compute the kron product part, similar to the
            % kron_b_nk/kron_b_nk2 ,start with the last mode
            
            %use U_mean_post_old & U_var_post_old here t get parallel
            %update
            bi = U_mean_post{nmod}(data(:,nmod),:);
            bi2 = U_var_post{nmod}(:,:,data(:,nmod))+ outer_prod_rows(bi);
            
            for j=nmod-1:-1:1
                mod=j;
                cur_term = U_mean_post{mod}(data(:,mod),:);
                bi = kron_prod_rows_vec(bi,cur_term);
                bi2 = kron_prod_rows_mat(bi2,U_var_post{mod}(:,:,data(:,mod))+ outer_prod_rows(cur_term));
            end
            % bi final shape: n* R_prod
            % bi2 final shape: R_prod * R_prod * n
            
            alpha = W_post_mean; %size: R_prod*1
            eta = W_post_var; %size: R_prod*1
            
            W_vec_var_post = inv(diag(1.0./eta) + tau*sum(bi2,3));%size: R_prod * R_prod : prior+llk
            W_vec_mean_post =  W_vec_var_post*(alpha./eta + bi'*y*tau);%size: R_prod * 1 : prior+llk
            
            
            W_vec_fac_var_post = inv( tau*sum(bi2,3));%size: R_prod * R_prod :llk factor
            W_vec_fac_mean_post = W_vec_fac_var_post*(bi'*y*tau);%size: R_prod * 1 : llk factor
            
 
        
            % update a&b for tau
            
            a_post = a_post_old + 0.5*n;
            b_post = b_post_old + 0.5*sum(y.^2);
           
            term1 = sum(y.*(bi*W_vec_mean_post));
            
            % 0.5*E(vec(W))*bi2*E(vec(W))
            temp = ttm(ttm(tensor(bi2),W_vec_mean_post',1),W_vec_mean_post',2);
            temp = squeeze(temp.data);
            term2 = 0.5*sum(temp);
            
            b_post = b_post - term1 + term2;
            
            tau = a_post/b_post;
            
      end
            if prior_update>0
                % update w & sparse parts
                W_mean_cal = W_vec_fac_mean_post; %W_vec_fac_mean_post; %size: R_prod*1
                W_var_cal = abs(diag(W_vec_fac_var_post));%abs(diag(W_vec_fac_var_post)); %size: R_prod*1

                % foumula(24)-(30) in draft
                test_var1 = elewise_normpdfln(W_mean_cal,zeros(size(W_mean_cal)),sqrt(W_var_cal+sigma_0^2));
                test_var2 = elewise_normpdfln(W_mean_cal,zeros(size(W_mean_cal)),sqrt(W_var_cal));
                rho_star = test_var1 - test_var2;
                %rho_star = mvnormpdfln(W_mean_cal,zeros(size(W_mean_cal)),diag(sqrt(W_var_cal+sigma_0)))-mvnormpdfln(W_mean_cal,zeros(size(W_mean_cal)),diag(sqrt(W_var_cal)));

                rho_cave = rho_star + inverse_sigmoid(rho_0);
                v_cav = 1.0./(1.0./W_var_cal + 1/sigma_0^2);
                mu_cav = v_cav.*(W_mean_cal./W_var_cal);

%                 W_post_mean = sigmoid_my(rho_cave).*mu_cav;
%                 W_post_var = sigmoid_my(rho_cave).*(v_cav + (1-sigmoid_my(rho_cave)).*(mu_cav).^2);
%                 W_post_rho = rho_star;
                
                W_post_mean_new = sigmoid_my(rho_cave).*mu_cav;
                W_post_var_new = sigmoid_my(rho_cave).*(v_cav + (1-sigmoid_my(rho_cave)).*(mu_cav).^2);
                
                % damping
                W_post_var = 1.0./(damping_alpha./W_post_var_old + (1-damping_alpha)./W_post_var_new);
                W_post_mean = W_post_var.*(damping_alpha.*W_post_mean_old./W_post_var_old + (1-damping_alpha).* W_post_mean_new./W_post_var_new);
                W_post_rho = damping_alpha.*W_post_rho_old +  (1-damping_alpha).*rho_star;
                
                
               
            end
            
        % test part
        test_len = size(test_X,1);
        bi = U_mean_post{nmod}(test_X(:,nmod),:);
        for j=nmod-1:-1:1
            mod=j;
            cur_term = U_mean_post{mod}(test_X(:,mod),:);
            bi = kron_prod_rows_vec(bi,cur_term);
        end
        pred_result  = bi*W_vec_mean_post; %size: n_test,1
        mse = sum((test_Y-pred_result).^2)/test_len;
        
        diff = 0;
        for k=1:nmod
            diff = diff + sum(sum(abs(old_u{k} - U_mean_post{k}(uind{k},:))))/sum(sum(abs(old_u{k})));
        end
        
        fprintf('iter = %d, diff = %g, rmse = %g, tau = %g\n', iter, diff,sqrt(mse),tau);
%         if diff< opt.tol
%             break;
%         end
        sparse_ratio = sum(W_post_rho<0)/prod(R_list);
        fprintf('sparse_ratio %g \n',sparse_ratio);
        
   
    
    post={U_mean_post,U_var_post,W_vec_mean_post,W_vec_var_post,W_post_rho,W_post_mean,W_post_var,[a_post,b_post]};
end