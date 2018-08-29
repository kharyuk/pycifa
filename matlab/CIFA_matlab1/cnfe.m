function [ nc nQ ] = cnfe( c, Q,opts )
% This function will extract common nonnegative features by solving
%  min \sum_n ||c*Q{n}-nc*nQ||
%   s.t. nc>=0  nQ>=0
% Usage: [ nc nQ ] = cnfe( c, Q,opts );
% 

defopts=struct('NumOfComp',[],'maxiter',2000,'tol',1e-6,'sparse_para',0,'ncheck',10);
if ~exist('opts','var')
    opts=struct();
end
[r maxiter tol alpha ncheck]=scanparam(defopts,opts);

if isempty(r)||r>size(c,2)
    r=size(c,2);
end
[M rc]=size(c);

nc=rand(M,r);
N=numel(Q);
for n=1:N
    nQ{n}=max((nc\c)*Q{n},1e-3);
end
NMFALG='mu';


switch lower(NMFALG)
    case 'mu'
        for it=1:maxiter
            %% update nc first
            nc0=nc;
            num=zeros(rc,r);
            denominator=zeros(r,r);
            
            for n=1:N
                num=num+(Q{n}*nQ{n}');
                denominator=denominator+(nQ{n}*nQ{n}');
            end
            for it=1:5
                nc=nc.*(max(c*num,eps)./max(nc*denominator,eps));
            end
            nc=bsxfun(@rdivide,nc,max(sum(nc),eps));
            
            %% update nQ{n} for each n
            nctc=nc'*c;
            nctnc=nc'*nc;
            for n=1:N
                for it=1:5
                    nQ{n}=nQ{n}.*(max((nctc)*Q{n},eps)./max((nctnc)*nQ{n}+alpha*nQ{n},eps));
                end
            end
            
            %% stopping criterion
            if (it>maxiter*0.25)&&(~rem(it,ncheck))
                if norm(nc-nc0,'fro')<tol
                    break;
                end
            end

        end
    otherwise
        error('Unsupported algorithm.');
end

end

