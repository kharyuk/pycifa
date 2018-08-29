import numpy as np

def mmc_nonnegative(A, Qini, iters=1000, tol=1e-8):
    # Min-Max cut with nonnegative relaxization
    # A: similarity matrix of graph, n*n matrix
    # Qini: initial cluster indicator matrix, n*c matrix
    # Q: output cluster indicator matrix, n*c matrix
    # Ref: 
    # Feiping Nie, Chris Ding, Dijun Luo, Heng Huang. 
    # Improved MinMax Cut Graph Clustering with Nonnegative Relaxation.  
    # The European Conference on Machine Learning and Principles and Practice of
    # Knowledge Discovery in Databases (ECML PKDD), Barcelona, 2010.
    class_num = Qini.shape[1]
    eps = np.spacing(1)
    D = np.sum(A, axis=1)
    Q = Qini.copy()
    # symmetrize 2
    obj = np.zeros(iters)
    obj1 = np.zeros(iters)
    orobj = np.zeros(iters)
    prevQ = Q.copy()
    for it in xrange(iters):
        dA = (1.+eps)/(np.diag(np.dot(Q.T, np.dot(A, Q))) + eps) #
        tmp1 = Q.T * D #
        QQ = np.dot(tmp1, Q)
        dD = np.diag(QQ)
        Qb = Q * (dA**2.)
        tmp2 = np.dot(A, Qb)
        Lambda = np.dot(Q.T, tmp2)
        Lambda = 0.5*(Lambda + Lambda.T)
        S = (tmp2 + eps) / (np.dot(tmp1.T, Lambda) + eps)
        S = S**0.5
        Q = Q*S
        tmp = Q.T*D
        Q = Q*np.sqrt((1.+eps)/(np.diag(np.dot(tmp, Q)) + eps))
        QQI = QQ - np.eye(class_num)
        obj[it] = sum(dA) - np.trace(np.dot(Lambda, QQI))
        obj1[it] = sum(dD*dA)
        orobj[it] = np.sqrt(np.trace(np.dot(QQI.T, QQI) / (class_num*(class_num-1))))   #### or class_num*(class_num+1) ?
        if np.linalg.norm(Q - prevQ) < tol:
            break
        prevQ = Q.copy()
    return Q, obj, obj1, orobj

