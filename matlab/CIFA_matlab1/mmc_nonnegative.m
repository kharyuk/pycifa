% Min-Max cut with nonnegative relaxization
function [Q obj obj1 orobj] = mmc_nonnegative(A, Qini)
% A: similarity matrix of graph, n*n matrix
% Qini: initial cluster indicator matrix, n*c matrix
% Q: output cluster indicator matrix, n*c matrix

% Ref: 
% Feiping Nie, Chris Ding, Dijun Luo, Heng Huang. 
% Improved MinMax Cut Graph Clustering with Nonnegative Relaxation.  
% The European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), Barcelona, 2010.



ITER = 1000;
class_num = size(Qini,2);

D = diag(sum(A,2));
Q = Qini;

% symmetrize 2
obj = zeros(ITER,1);
obj1 = zeros(ITER,1);
orobj = zeros(ITER,1);
for iter = 1:ITER
    dA = 1./diag(Q'*A*Q);
    QQ = Q'*D*Q;
    dD = diag(QQ);
    Qb = Q*diag((dA.^2));
    Lamda = Q'*A*Qb;
    Lamda = (Lamda + Lamda')/2;

    S = (A*Qb + eps)./(D*Q*Lamda + eps);
    S = S.^(1/2);
    Q = Q.*S;
    Q = Q*diag(sqrt(1./diag(Q'*D*Q)));
    
    QQI = QQ - eye(class_num);
    obj(iter) = sum(dA) - trace(Lamda*(QQI));
    obj1(iter) = sum(dD.*dA);
    orobj(iter) = sqrt(trace(QQI'*QQI)/(class_num*(class_num-1)));
end;




