%% This code implement the M-CCA algorithm based on the SSQCOR cost 
%% Reference:
%% J. R. Kettenring, “Canonical analysis of several sets of variables,?
%% Biometrika, vol. 58, pp. 433?1, 1971.

%% Input: 
%% y: M by 1 cell array containing the *prewhitened* group datasets
%% numOfCV: Number conanical vectors to be estimated
%% B0: M by 1 cell array containing the initial guess of the demixing
%% matrices for the group dataset: default is identity matrix
%% Output:
%% B: M by 1 cell array containing the estimated demixing matrices 
%% theta_opt: Vector containing cost function values at the optimal
%% solutions

%% Yiou (Leo) Li Mar. 2009

function [B, theta_opt] = ssqcor_cca_efficient(y, numOfCV, B0)

numMaxIter = 1000;
M = length(y);
[p, N] = size(y{1});
eps = 0.0001;

%% Calculate all pairwise correlation matrices
for i = 1:M
    for j = i:M
        R{i,j} = y{i}*y{j}'/N;
        R{j,i} = R{i,j}';
    end
    R{i,i} = eye(p);
end

%% Obtain a prilimary estimate by MAXVAR algorithm
if nargin < 3
%     B0 = maxminvar_cca(y, numOfCV);
    for i = 1:M
        B0{i} = eye(numOfCV,p);
    end
end
clear y;
B = cell(M,1);

for s = 1:numOfCV    
    
    %% Iterations to solve the s-th stage canonical vectors for 1--M
    %% datasets
    theta_old = 0; % zeros(M,1);
    theta = 0; % zeros(M,1);
    for n = 1:numMaxIter     
        
        if n == 1
            %% Initialize B{1--M}(s,:) by B0;
            for j = 1:M
                B{j}(s,:) = B0{j}(s,:)/norm(B0{j}(s,:));  %% Use normalized B0
            end
            %% Calculate the cost funtion at the initial step
            for j = 1:M
                for k = 1:M
                    R_hat(j,k) = B0{j}(s,:)*R{j,k}*B0{k}(s,:)';
                end
            end
            theta_0 = trace(R_hat*R_hat');
        end
        
        %% Solve the current canonical vector for the j-th dataset
        for j = 1:M
            
%             %% Calculate the cost function
%             jtheta_old(j) = 0;
%             for k = 1:M
%                 jtheta_old(j) = jtheta_old(j) + B{k}(s,:)*R{k,j}*B{j}(s,:)';
%             end

            %% Calculate the terms for updating jbn
			jC = B{j}(1:s-1,:)';
            if s ~= 1
                jA = eye(p) - jC*jC'; % *inv(jC'*jC)
            else 
                jA = eye(p);
            end
            jP = 0;
            for k = 1:M
                if k ~= j
                    jP = jP + R{j,k}*B{k}(s,:)'*B{k}(s,:)*R{k,j};
                end
            end
            %% update jbn
            [Ev Dv] = eig(jA*jP);
            DD = diag(Dv);
            [maxv maxi] = max(DD);
            B{j}(s,:) = Ev(:,maxi)';
            tmp(j) = Dv(maxi,maxi) + 1; % should = jtheta(j)
            
%             %% Calculate the cost function
%             jtheta(j) = 0;
%             for k = 1:M
%                 jtheta(j) = jtheta(j) + B{k}(s,:)*R{k,j}*B{j}(s,:)';
%             end
%             chec(j) = tmp(j) - jtheta(j);
%             delta(j) = jtheta(j) - jtheta_old(j);
        end
        
        %% Calculate the cost funtion at the current step
        for j = 1:M
            for k = 1:M
                R_hat(j,k) = B{j}(s,:)*R{j,k}*B{k}(s,:)';
            end
        end
        theta(n) = trace(R_hat*R_hat');
        
        %% Check termination condition
        if sum(abs(theta(n) - theta_old)) < eps | n == numMaxIter
            theta_opt(s) = theta(n);
            break;
        end
        theta_0;
        theta_old = theta(n);
    end
    
%     fprintf('\n Component #%d is estimated, in %d iterations',s, n);
   
end

return;

