function [S theta_opt]=call_mcca(X_all,numOfCV)
%% call ssqcor_cca
% 

%% transpose the input
X_all=cellfun(@(x) x',X_all,'uni',false);

%% pre-processing
y=preprocess_mcca(X_all);

%% call MCCA
[B, theta_opt] = ssqcor_cca_efficient(y, numOfCV);

%% Output
N=numel(y);
S=cell(N,1);
for n=1:N
    S{n}=(B{n}*y{n})';
end




function group_pc_org=preprocess_mcca(X_all)
%% Pre-processing before joint-BSS: PCA + sphering
%  Extracted from exp_cca_multiset_bss_robustness_public.m
%   by G. Zhou
K=numel(X_all);
numPCv = cell(K,1);
B = cell(K,1);
for i = 1:K
    mixedsig = X_all{i};
    numPCv{i} = size(mixedsig,1);
    %% PCA
    out=mixedsig*mixedsig'/size(mixedsig,2);
    [V,D] = eig(out);                  
    [eigenval,index] = sort(diag(D));
    index=flipud(index);
    EigenValues=flipud(eigenval)';
    EigenVectors=V(:,index);
    data = EigenVectors'*mixedsig;
    
    %% Sphering
    sphere_E = inv(diag(sqrt(EigenValues(1:numPCv{i}))));
    data = sphere_E*data(1:numPCv{i},:);
    %% Save PCA + sphering results
    B{i} = EigenVectors(:,1:numPCv{i})*inv(sphere_E);
    %% Pass the whitened data to 'group_pc_org' for M-CCA
    group_pc_org{i} = data(1:numPCv{i},:);
end