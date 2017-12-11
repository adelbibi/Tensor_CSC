function [y, objV] = prox_111_norm(x,lambda,rho) 
% Proximal function of 1,1,1,1 tensor norm


s = pos( 1 - lambda./(rho*abs(x)) );
y = s .*  x;

objV = sum(sum( sum(y.*y,3)  ));


end

