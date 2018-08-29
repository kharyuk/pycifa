function [ nc, nQ ] = cnfe( c, Q,opts )
% This function will extract common nonnegative features by solving
%  min \sum_n ||c*Q{n}-nc*nQ{n}||
%   s.t. nc>=0  nQ>=0
% Usage: [ nc nQ ] = cnfe( c, Q,opts );
% 
maxiniter=5;
NMFALG='hals';

trackit=10;
defopts=struct('NumOfComp',[],'maxiter',500,'maxiniter',20,'tol',1e-6,'sparse_para',1e-12,'ncheck',10);
if ~exist('opts','var')
    opts=struct();
end
[r, maxiter, maxiniter, tol, sparse_para, ncheck]=scanparam(defopts,opts);

if isempty(r)||r>size(c,2)
    r=size(c,2);
end
[M, rc]=size(c);

N=numel(Q);
nc=rand(M,r);
nQ=cell(1,N);
for n=1:N
    nQ{n}=max((nc\c)*Q{n},1e-3);
end


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
            for init=1:maxiniter
                nc=max(nc.*(c*num-sparse_para)./max(nc*denominator,eps),eps);
            end
            nc=bsxfun(@rdivide,nc,max(sum(nc),eps));
            
            %% update nQ{n} for each n
            nctc=nc'*c;
            nctnc=nc'*nc;
            for n=1:N
                for init=1:maxiniter
                    nQ{n}=nQ{n}.*(max((nctc)*Q{n},eps)./max((nctnc)*nQ{n},eps));
                end
            end
            
            %% stopping criterion
            if (it>maxiter*0.25)&&(~rem(it,ncheck))
                if norm(nc-nc0,'fro')<tol
                    break;
                end
            end

        end
    case 'hals'
        
        for it=1:maxiter
                %% update nQ{n}
            wtw=nc'*nc;
            nrm2w=max(diag(wtw),eps);
            for n=1:N
                wtxy=(nc'*c)*Q{n};
                for init=1:maxiniter
                    od=randperm(rc);
                    for i=od
                        nQ{n}(i,:)=max(nQ{n}(i,:)*nrm2w(i)+(wtxy(i,:)-wtw(i,:)*nQ{n}),eps)/nrm2w(i);
                    end % for col
                end % for init
            end % for n
            
            %% update nc
            nc0=nc;
            yht=zeros(rc,rc);
            hht=zeros(rc,rc);
            for n=1:N       
                yht=yht+Q{n}*nQ{n}';
                hht=hht+nQ{n}*nQ{n}';
            end
            xyht=c*yht;
            nrm2h=max(diag(hht),eps);
            for init=1:maxiniter
                od=randperm(rc);
                for i=od
                    nc(:,i)=(nc(:,i)*nrm2h(i)+xyht(:,i)-nc*(hht(:,i)))./(nrm2h(i));
                    nc(:,i) = max(nc(:,i)-sparse_para,eps);
                end
            end
            
            
            if (trackit>0)&&(~rem(it,trackit))
                if norm(nc-nc0,'fro')<tol
                    break;
                end
            end
        end
        
    otherwise
        error('Unsupported algorithm.');
end

end

