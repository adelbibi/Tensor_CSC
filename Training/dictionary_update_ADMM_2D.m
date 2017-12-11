function [D,error_DTnorm,error_reg] = dictionary_update_ADMM_2D(Dhat,Xhat,Yhat,n1,n3,n4,K,N,filter_szx,filter_szy)
That = Dhat;
Ghat = That;
%% ADMM updates parameters init
rho = 1;
gamma = 1e-2;
rho_max = 600;
error_DTnorm_thresh = 1e-7;
error_reg_change_thresh = 1e-7;
error_DTnorm = inf;
max_iter = 10;
counter = 0;
temp2=[];
error_reg = [];
Dhat_Cat = [];
counter_error = 1;
%% ADMM
while(true)
    counter = counter + 1;
    if(counter > max_iter)
        break;
    end
    %% Update Dhat
    parfor comb_iind=1:(n3*n4)
        rhs = (Yhat(:,:,comb_iind)*Xhat(:,:,comb_iind)' + rho*That(:,:,comb_iind) - Ghat(:,:,comb_iind));
        if(N==1)
            Dhat_Cat(:,:,comb_iind) = sherman_morrison_inv(rhs,Xhat(:,:,comb_iind),rho);
        else
            Dhat_Cat(:,:,comb_iind) = rhs / (Xhat(:,:,comb_iind)*Xhat(:,:,comb_iind)' + rho*eye(K));
        end
    end
    Dhat = reshape(Dhat_Cat,n1,K,n3,n4);
    %% Update That
    temp = Dhat + (1/rho)*Ghat;
    That = prox_proj_norm(temp,filter_szx,filter_szy,n1,n3,n4);
    %% Update Ghat
    Ghat = Ghat + rho * (Dhat - That);
    %% Compute cost and errors
    if(mod(counter,10) == 0)
        D_T_errors = reshape(Dhat - That,[],1);
        error_DTnorm(counter)  = sqrt(D_T_errors'*D_T_errors);
        for image_train=1:N
            parfor comb_ind_kw=1:(n3*n4)
                temp2(:,image_train,comb_ind_kw) = (Yhat(:,image_train,comb_ind_kw) - Dhat(:,:,comb_ind_kw)*Xhat(:,image_train,comb_ind_kw));
            end
        end
        for image_train=1:N
            du22mmy = temp2(:,image_train,:);
            dummy_sum(image_train) = sqrt(du22mmy(:)'*du22mmy(:));
        end
        error_reg(counter) = sum(dummy_sum);
        if (counter == 1)
            error_reg_change = 0;
            error_DTnorm_change = 0;
        else
            error_reg_change = norm(error_reg(end) - error_reg(end-1));
            error_DTnorm_change = norm(error_DTnorm(end) - error_DTnorm(end-1))/norm(error_DTnorm(end-1));
        end
        counter_error = counter_error + 1;
        %% Print
        if mod(counter,1)== 0
            fprintf('+ Iter: %f  RegError: %1.3f ConsError: %1.3f Rho: %f \n',counter,error_reg(end),error_DTnorm(end),rho);
        end
        %% Checks for breaks
        if(counter_error > 2)
            if(error_DTnorm_change < error_DTnorm_thresh || error_reg_change < error_reg_change_thresh)
                break;
            end
        end
    end
    %% Parameter update
    rho = rho*(1+gamma);
    rho = min(rho_max, rho);
end
%% Sol
Dhat_per = permute(Dhat,[3,4, 1, 2]);
D_per = real(ifft2(Dhat_per));
D = permute(D_per,[3,4, 1, 2]);
fprintf('+ Updateing D (Dictionary Learning): took %d iterations. \n',counter);
end

