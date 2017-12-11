function [D3] = sherman_morrison_inv(B,x,rho)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

% Solve DY = B ---> D = B * (Y)^-1
dummy = B*x;
normalized_val = 1/(x'*x + rho);
D3 = (1/rho) * (B - (normalized_val*dummy*x')  );


end

