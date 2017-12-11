function [X,error_XZnorm,error_reg] = sparse_code_update_ADMM_2D(Dhat,Xhat,Yhat,n3,n4,K,N,lambda)
Xhat_per = permute(Xhat,[3,4, 1, 2]);
X_per = real(ifft2(Xhat_per))*sqrt(n3*n4);
X = permute(X_per,[3,4, 1, 2]);
Z = X;
U = Z;
%% Conj function parameters
pcg_tol = 1e-7;
%% ADMM updates parameters init
rho = 1;
gamma = 1e-2;
rho_max = 600;
error_XZnorm_thresh = 1e-7;
error_reg_change_thresh = 1e-7;
error_XZnorm = inf;
max_iter = 150;
counter = 0;
temp2=[];
error_reg = [];
Xhat_Cat=[];
counter_error = 1;
%% ADMM
while(true)
    counter = counter + 1;
    if(counter > max_iter)
        break;
    end
    %Prepare data for foureir domain solution of X
    Z_per = permute(Z,[3, 4, 1, 2]); Zhat_per = fft2(Z_per)/sqrt(n3*n4); Zhat = permute(Zhat_per,[3, 4, 1, 2]);
    U_per = permute(U,[3, 4, 1, 2]); Uhat_per = fft2(U_per)/sqrt(n3*n4); Uhat = permute(Uhat_per,[3, 4, 1, 2]);
    %% Update Xhat in the Foureir Domain
    for image_train=1:N
        parfor comb_ind=1:(n3*n4)
            rhs = (Dhat(:,:,comb_ind)'*Yhat(:,image_train,comb_ind) + rho*Zhat(:,image_train,comb_ind) - Uhat(:,image_train,comb_ind));
            [Xhat_Cat(:,image_train,comb_ind),cg_flag,~,pcg_iter] = pcg(@afun_Xhat,rhs,pcg_tol,[],[],[],Xhat(:,image_train,comb_ind),Dhat(:,:,comb_ind),rho);
        end
    end
    Xhat = reshape(Xhat_Cat,K,N,n3,n4);
    X_hat_per = permute(Xhat,[3,4,1,2]);
    %going back to time domain
    X = permute(real(ifft2(X_hat_per)),[3,4,1,2])*sqrt(n3*n4);
    %% Update Zhat
    temp = X + (1/rho)*U;
    Z = prox_111_norm(temp,lambda,rho);
    %% Update Uhat
    U = U + rho * (X - Z);
    %% Compute cost and errors
    if(mod(counter,10) == 0)
        X_Z_errors = reshape(X - Z,[],1);
        error_XZnorm(counter_error)  = sqrt(X_Z_errors'*X_Z_errors);
        for image_train=1:N
            parfor comb_ind_kw=1:(n3*n4)
                temp2(:,image_train,comb_ind_kw) = (Yhat(:,image_train,comb_ind_kw) - Dhat(:,:,comb_ind_kw)*Xhat(:,image_train,comb_ind_kw));
            end
        end
        for image_train=1:N
            du22mmy = temp2(:,image_train,:);
            dummy_sum(image_train) = sqrt(du22mmy(:)'*du22mmy(:));
        end
        error_reg(counter_error) = sum(dummy_sum);
        if (counter_error == 1)
            error_reg_change = 0 ;
            error_XZnorm_change = 0;
        else
            error_reg_change = norm(error_reg(end) - error_reg(end-1))/norm(error_reg(end-1));
            error_XZnorm_change = norm(error_XZnorm(end) - error_XZnorm(end-1))/norm(error_XZnorm(end-1));
        end
        counter_error = counter_error + 1;
        %% Print
        if mod(counter,1)== 0
            fprintf('+ Iter: %f  RegError: %1.3f ConsError: %1.3f Rho: %f \n',counter,error_reg(end),error_XZnorm(end),rho);
        end
        %% Checks for breaks
        if(counter_error > 2)
            if(error_XZnorm_change < error_XZnorm_thresh || error_reg_change < error_reg_change_thresh)
                break;
            end
        end
    end
    %% Parameter update
    rho = rho*(1+gamma);
    rho = min(rho_max, rho);
end
Xhat_per = permute(Xhat,[3,4, 1, 2]);
X_per = real(ifft2(Xhat_per));
X = permute(X_per,[3,4, 1, 2]);
fprintf('+ Updateing X (Sparse Code): took %d iterations. \n',counter);
return;
%% functions for variable update
function [res] = afun_Xhat(Xhat,Dhat,rho)
dummy_1 = Dhat* Xhat;
dummy_2 = Dhat'*dummy_1;
res = dummy_2 + rho*Xhat;
return;