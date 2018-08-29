function [J,A] = JIVE(Datatypes,r,rIndiv,scale,ConvergeThresh,MaxIter)
% [J,A] = JIVE(Datatypes,r,rIndiv,scale,ConvergeThresh,MaxIter)
%Input:
%   - Datatypes: Cell array of data matrices with matching columns
%   (Datatypes{i} gives the i'th data matrix)
%   - r: Given rank for joint structure
%   - rIndiv: Given vector fo ranks for individual structure'
%   - scale: Should the datasets be centered and scaled by total variation?
%   (default = 'y')
%   - ConvergeThresh:  Convergence threshold (default = 10^(-10))
%   - MaxIter: Maximum numeber of iterations (default = 1000)
%Output:
%   - J: Matrix of joint structure, of dimension (d1+d2+...+dk, n)
%   - A: Matrix of individual structure
if(class(Datatypes)~='cell')
    disp('Error: Datatypes should be of class ''cell''');
    disp('For data matrices X1,X2,...,Xk, input Datatypes = {X1,X2,...,Xk}');
end
nDatatypes = length(Datatypes);

for i=1:nDatatypes
    [d(i),n(i)] =size(Datatypes{i});
end
n = unique(n);
if(length(n) ~= 1)
    disp('Error: Datatypes do not have same number of columns');
end

if(nargin<4)
    scale = 'y';
end

if(nargin<5)
   ConvergeThresh = 10^(-10);
end
if(nargin<6)
   MaxIter = 1000;
end

if(scale =='y')
    for(i=1:nDatatypes)
        Datatypes{i} = bsxfun(@minus,Datatypes{i},mean(Datatypes{i}')');
        Datatypes{i} = Datatypes{i}/norm(Datatypes{i}, 'fro');
    end
end
       
%Dimension reducing transformation for high-dimensional data
U_original = cell(1,nDatatypes);
for(i=1:nDatatypes)
    if(d(i) > n)
        [U,S,V] = svds(Datatypes{i},n-1);
        Datatypes{i} = S*V';
        [d(i),n] = size(Datatypes{i});
        U_original{i} = U;
    end
end

Tot = zeros(sum(d),n);
for(i = 1:(nDatatypes))
    Tot((sum(d(1:(i-1)))+1):sum(d(1:i)),:) = Datatypes{i};
end
J = zeros(sum(d),n);
A = zeros(sum(d),n);
X_est = 0;
for(j=1:MaxIter)
    [V1,S,V2] = svds(Tot,r);
    J = V1*S*V2';
    for(i=1:nDatatypes)
        rows =(sum(d(1:(i-1)))+1):sum(d(1:i));
        U = Datatypes{i} - J(rows,:);
        [UV1,US,UV2] = svds(U-U*V2*V2',rIndiv(i));
        A(rows,:) = UV1*US*UV2';
        Tot(rows,:) = Datatypes{i}-UV1*US*UV2';
    end
    if(norm(X_est-J-A,'fro')^2 < ConvergeThresh)
        break
    end
    X_est = J+A;
    if(j==MaxIter)
        disp('Warning: MaxIter iterations reached before convergence');
    end
end

%Transform back to original space
J_original = [];
A_original = [];
for(i=1:nDatatypes)
Joint = J((sum(d(1:(i-1)))+1):sum(d(1:i)),:);
Indiv = A((sum(d(1:(i-1)))+1):sum(d(1:i)),:);
    if(~isempty(U_original{i}))
        Joint = U_original{i}*Joint;
        Indiv = U_original{i}*Indiv;
    end
    J_original = [J_original; Joint];
    A_original = [A_original; Indiv];
end

J = J_original;
A = A_original;



