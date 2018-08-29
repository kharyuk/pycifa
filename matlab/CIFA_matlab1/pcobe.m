function [Ac Bc res Zi] = pcobe( Y,opts )
% Common orthogonal basis extraction
% Usage: [
defopts=struct('c',[],'maxiter',2000,'PCAdim',0.8,'tol',1e-6,'epsilon',0.03,'pdim',[]);
if ~exist('opts','var')
    opts=struct;
end
[c maxiter PCAdim tol epsilon pdim]=scanparam(defopts,opts);

Ydim=cellfun(@(x) size(x,1),Y);
NRows=size(Y{1},1);
N=numel(Ydim);
if ~all(Ydim==NRows)
    error('Y must have the same number of rows.');
end

if isempty(pdim)
    pdim=max(cellfun(@(x) size(x,2),A));
    pdim=min(NRows,pdim);
end
P=randn(pdim,NRows);


U=cell(1,N);
J=zeros(1,N);
PY=cell(1,N);
for n=1:N
    PY{n}=P*Y{n};
    U{n}=orth(PY{n});    
    J(n)=size(U{n},2);
end


%% some matrices are of full-row rank.  Pre-processing by using PCA is required.
flag=J(n)>pdim;
if sum(flag)>1
    if PCAdim<1
        rdims=max(floor(pdim*PCAdim),2);
    else
        rdims=min(pdim,PCAdim);
    end
    for n=1:N
        if flag(n)
            [U{n} d v]=svds(PY{n},rdims,'L');
            J(n)=rdims;
        end
    end
end



%% Seeking the first common basis
Ac=U{1}*randn(J(1),1);
Ac=Ac./norm(Ac);

x=cell(1,N);
for it=1:maxiter
    c0=Ac;
    c1=zeros(pdim,1);
    for n=1:N
        x{n}(:,1)=U{n}'*Ac;
        c1=c1+U{n}*x{n}(:,1);
    end
    Ac=c1./norm(c1,'fro');
    if abs(c0'*Ac)>1-tol
        break;
    end
end


res(1)=0;
for n=1:N
    temp=U{n}'*Ac;
    res(1)=res(1)+1-temp'*temp;
end
res(1)=res(1)/N;


if res(1)>epsilon&&isempty(c)  %% c is not specified. 
    disp('No common basis found.');
    Ac=[];    Bc=Y;    Yi=Y;
    return;
end


minJn=min(J);

if ~isempty(c)
    c=min([c,J]);
    res=[res inf(1,c-1)];
    Ac=[Ac zeros(pdim,c-1)];
else
    res=[res inf(1,minJn-1)];
    Ac=[Ac zeros(pdim,minJn-1)];
end

%% seekign the other common basis
for j=2:minJn    
    %% %% stopping criterion -- 1 where c is given
    if ~isempty(c)&&(j>c)
        break;
    end
    
    %% update U;
    for n=1:N
        U{n}=U{n}-(U{n}*x{n}(:,j-1))*x{n}(:,j-1)';
    end
    Ac(:,j)=U{1}*randn(size(U{1},2),1);
    Ac(:,j)=Ac(:,j)./norm(Ac(:,j),'fro');
    
    %% get another column
    for it=1:maxiter
        c0=Ac(:,j);
        c1=zeros(pdim,1);
        for n=1:N
            x{n}(:,j)=U{n}'*Ac(:,j);
            c1=c1+U{n}*x{n}(:,j);
        end        
                
        Ac(:,j)=c1./norm(c1,'fro');
        
        if abs(c0'*Ac(:,j))>1-tol
            break;
        end
        
    end % it 
    
    res(j)=0;
    for n=1:N
        temp=U{n}'*Ac(:,j);
        res(j)=res(j)+1-temp'*temp;
    end
    res(j)=res(j)./N;
     
    %% stopping criterion -- 2
    if res(j)>epsilon&&isempty(c)
        res(j)=inf;      
        break;
    end    
    
end % each j
% res
flag=isinf(res);
res(flag)=[];
Ac(:,flag)=[];

%% proj
cP=zeros(NRows,c);
for n=1:N
    x{n}=PY{n}\Ac;
    cP=cP+Y{n}*x{n};
end
Ac=orth(cP);

Zi=cell(1,N);
if nargout>=2
    Bc=cell(1,N);
    Zi=cell(1,N);
    for n=1:N
        Bc{n}=(Ac'*Y{n});
        Zi{n}=Y{n}\Ac;
    end
end

end

