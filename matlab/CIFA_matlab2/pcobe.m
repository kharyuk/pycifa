function [Ac, Bc, f, Zc] = pcobe( Y,opts )
% Common orthogonal basis extraction

if isfield(opts,'Lp')
    Lp=opts.Lp;
    opts=rmfield(opts,'Lp');
else
    Lp=[];
end

Ydim=cellfun(@(x) size(x,1),Y);
NRows=size(Y{1},1);
N=numel(Ydim);
if ~all(Ydim==NRows)
    error('Y must have the same number of rows.');
end

if isempty(Lp)
    Lp=floor(Ydim*0.5);
end
if Lp<max(cellfun(@(x) size(x,2),Y))
    error('Lp is too small. Try a larger one.');
end

if Lp>=Ydim
    error('Lp is too large. Use a smaller value or call cobe/cobec directly.');
end


P=randn(Lp,NRows);

PY=cell(1,N);
for n=1:N
    PY{n}=P*Y{n};
end

[Ac,Bc,f,Zc]=cobe(PY,opts);
c=size(Ac,2);


if nargout>=2
    Ac=zeros(NRows,c);
    for n=1:N
        Ac=Ac+Y{n}*Zc{n};
    end
    Ac=Ac*(Ac'*Ac)^-.5;
    Bc=cellfun(@(x) Ac'*x,Y,'uniform',false);
    
    for n=1:N
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

