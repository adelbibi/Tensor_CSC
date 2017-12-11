function [That] = prox_proj_norm(temp,filter_szx,filter_szy,n1,n3,n4)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here

temp_no_hat = real(ifft2(permute(temp,[3,4, 1, 2])));
temp_no_hat_small = temp_no_hat(1:filter_szx,1:1:filter_szy,:,:);
norm_matrix = sqrt(sum(sum(sum( conj(temp_no_hat_small).*temp_no_hat_small , 1),2),3)) ;
ind = norm_matrix>=1;
temp_no_hat_small(:,:,:,ind) = temp_no_hat_small(:,:,:,ind)./ repmat(norm_matrix(ind),filter_szx,filter_szy,n1,1,1);
temp_no_hat_small = padarray(temp_no_hat_small,[n3-filter_szx, n4-filter_szy],'post');
That = permute(fft2(temp_no_hat_small),[3 4 1 2]);

end

