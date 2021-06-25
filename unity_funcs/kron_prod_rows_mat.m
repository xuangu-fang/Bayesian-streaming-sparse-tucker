function Z = kron_prod_rows_mat(X,Y)
    [r1,r1,m] = size(X);
    [r2,r2,n] = size(Y);
    
    if m~=n
        error('size does not match');
    end
    
    Z = zeros(r1*r2,r1*r2,n); 
     
    for k =1:n
        Z(:,:,k) = kron(X(:,:,k),Y(:,:,k));
    end
end