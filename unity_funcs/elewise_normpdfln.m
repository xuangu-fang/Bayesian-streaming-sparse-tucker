function Z = elewise_normpdfln(X,Mean, Std)
n = length(Mean);

Z = zeros(n,1);

for k = 1:n
    Z(k) = mvnormpdfln(X(k),Mean(k),Std(k));
end 

end