function [Ac Bc Zi] = cobec( Y,opts )
%COMSSPACE Summary of this function goes here
%   Detailed explanation goes here
defopts=struct('c',1,'maxiter',200,'ctol',1e-3,'PCAdim',.8);
if ~exist('opts','var')
    opts=struct;
end
[c maxiter ctol]=scanparam(defopts,opts);

Ydim=cellfun(@(x) size(x,1),Y);
NRows=size(Y{1},1);
N=numel(Ydim);
if ~all(Ydim==NRows)
    error('Wrong dimension of Y.');
end

U=cell(1,N);
x=cell(1,N);
Ac=zeros(size(Y{1},1),c);
J=zeros(1,N);
for n=1:N
    U{n}=orth(Y{n});   
    J(n)=size(U{n},2);
    if size(U{n},2)<c
        error('Rank deficient and c is too large.');
    end
    
    if J(n)>NRows
        if PCAdim<1
            J(n)=max(NRows*PCAdim,2);
        else
            J(n)=min(NRows-1,PCAdim);
        end
        [U{n} d v]=svds(Y{n},J(n),'L');
    end
    
    %% initialize
    x{n}=randn(size(U{n},2),c);
    x{n}=orth(x{n});
    Ac=Ac+U{n}*x{n};
    [u , temp, v]=svds(Ac,c);
    Ac=u*v';
end

%% iterations
x=cell(1,N);
for it=1:maxiter
    c0=Ac;
    
    c2=zeros(NRows,c);
    for n=1:N
        x{n}=U{n}'*Ac;
        c2=c2+U{n}*x{n};
    end
    
%     [u , temp, v]=svds(c2,c,'L');
%     Ac=u*v';

    Ac=orth(c2);   
    
    %% stop
    if it>20&&mean(abs(diag(Ac'*c0)))>1-ctol
        break;
    end
end

if nargout>=2
    Bc=cell(1,N);
    Zi=cell(1,N);
    for n=1:N
        Bc{n}=(Ac'*Y{n});
        Zi{n}=Y{n}\Ac;
    end
end


end

