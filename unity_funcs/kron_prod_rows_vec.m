function Z = kron_prod_rows_vec(X,Y)
    [m,r1] = size(X);
    [n,r2] = size(Y);
    
    if m~=n
        error('size does not match');
    end
    
    Z = zeros(n,r1*r2); 
     
    for k =1:n
        Z(k,:) = squeeze(kron(X(k,:),Y(k,:)));
    end
end