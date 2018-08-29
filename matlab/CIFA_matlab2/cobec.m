function [Ac, Bc, f] = cobec( Y,opts )
%COMSSPACE Summary of this function goes here
%   Detailed explanation goes here
defopts=struct('c',1,'maxiter',500,'ctol',1e-3,'PCAdim',[]);
if ~exist('opts','var')
    opts=struct;
end
[C, maxiter, ctol,PCAdim]=scanparam(defopts,opts);

Ynrows=cellfun(@(x) size(x,1),Y);
Yncols=cellfun(@(x) size(x,2),Y);
NRows=size(Y{1},1);
N=numel(Ynrows);
if isempty(PCAdim)
    PCAdim=Yncols;
end

if ~all(Ynrows==NRows)
    error('Wrong dimension of Y.');
end

U=cell(1,N);

% x=cell(1,N);
% Ac=zeros(size(Y{1},1),C);

J=zeros(1,N);
for n=1:N
    if PCAdim(n)==Yncols(n)
%         [U{n}, R]=qr(Y{n},0);
%         flag=sum(R.^2,2)>1e-6;
%         U{n}=U{n}(:,flag);
        [E,D]=eig(Y{n}'*Y{n});
        D=diag(D);
        flag=D>1e-3;
        U{n}=Y{n}*E(:,flag)*diag(D(flag).^-.5);
    else
        U{n}=lowrankapp(Y{n},PCAdim(n),'pca');
    end
    J(n)=size(U{n},2);
    
    %% initialize
%     x{n}=randn(size(U{n},2),c);
%     Ac=Ac+U{n}*x{n};
end
% [u , ~, v]=svd(Ac,0);
% Ac=u*v';
[~,idx]=sort(J,'ascend');

Ac=ccak_init(U{idx(1)},U{idx(2)},C);
    
    
if any(J>=NRows)
    error('Rank of Y{n} must be less than the number of rows of {Yn}. You may need to specify the value for PCAdim properly.');
end


%% iterations
x=cell(1,N);
for it=1:maxiter
    c0=Ac;
    
    c2=zeros(NRows,C);
    for n=1:N
        x{n}=U{n}'*Ac;
        c2=c2+U{n}*x{n};
    end
   
    %% SVDS
%     [u , ~, v]=svds(c2,C,'L');
%     Ac=u*v';
%     [u,sigma]=eig(c2'*c2);
%     Ac=c2*u*(sigma^-.5*u');
    Ac=c2*(c2'*c2)^-.5;

%%  ORTH
%     Ac=orth(c2);   

%% qr
%     [Ac,temp]=qr(c2,0);
    
    %% stop
    if mean(abs(diag(Ac'*c0)))>1-ctol
        break;
    end
end


if nargout>=2
    Bc=cell(1,N);
    Zc=cell(1,N);
    for n=1:N
        Bc{n}=(Ac'*Y{n});
        
  
        [E,D]=eig(Y{n}'*Y{n});
        D=diag(D);
        flag=D>1e-6;
        E=E(:,flag)';
        D=D(flag);
        Zc{n}=E\(diag(D.^-1)*E*(Y{n}'*Ac));
    end
    
    f=zeros(1,size(Ac,2));
    for j=1:size(Ac,2)
        for n=1:N
            f(j)=f(j)+norm(Y{n}*Zc{n}(:,j)-Ac(:,j))^2;
        end
    end
    f=f./N;
end


end

