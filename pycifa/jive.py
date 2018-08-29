import numpy as np
import copy

from utils import fast_svd

#################### JIVE.m ####################################################

def JIVE(Datatypes_in, r, rIndiv, scale=True, ConvergeThresh=1e-10, MaxIter=1000):
    '''
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
    '''
    Datatypes = copy.deepcopy(Datatypes_in)
    assert isinstance(Datatypes, (tuple, list)), 'Error: Datatypes should be '+\
        'of class "tuple" or "list". For data matrices X1, X2, ..., Xk, input' +\
        ' Datatypes = {X1,X2,...,Xk}'
    nDatatypes = len(Datatypes)
    d = []
    n = []
    for i in xrange(nDatatypes):
        tmp = Datatypes[i].shape
        d.append(tmp[0])
        n.append(tmp[1])
    n = set(n)
    n = list(n)
    assert len(n) == 1, 'Error: Datatypes do not have same number of columns'
    n = n[0]
    if scale:
        for i in xrange(nDatatypes):
            Datatypes[i] -= np.mean(Datatypes[i], axis=1, keepdims=True) #########
            Datatypes[i] /= np.linalg.norm(Datatypes[i])
    # Dimension reducing transformation for high-dimensional data
    U_original = [None]*nDatatypes
    for i in xrange(nDatatypes):
        if (d[i] > n):
            U, S, Vt = fast_svd(Datatypes[i], n-1)
            Datatypes[i] = (Vt.T * S).T
            d[i], n = Datatypes[i].shape
            U_original[i] = U.copy()
    Tot = np.zeros([np.sum(d), n])
    for i in xrange(nDatatypes):
        ind1 = int(np.sum(d[:i]))
        ind2 = int(np.sum(d[:i+1]))
        Tot[ind1 : ind2, :] = Datatypes[i].copy()
    J = np.zeros([np.sum(d), n])
    A = np.zeros([np.sum(d), n])
    X_est = np.zeros([np.sum(d), n])
    for j in xrange(MaxIter):
        V1, S, V2t = fast_svd(Tot, r)
        J = np.dot(V1*S, V2t)
        for i in xrange(nDatatypes):
            ind1 = int(np.sum(d[:i]))
            ind2 = int(np.sum(d[:i+1]))
            rows = tuple(range(ind1, ind2))
            U = Datatypes[i] - J[rows, :]
            tmp = np.dot(U, V2t.T)
            tmp = np.dot(tmp, V2t)
            tmp = U - tmp
            UV1, US, UV2t = fast_svd(tmp, rIndiv[i])
            A[rows, :] = np.dot(UV1*US, UV2t)
            Tot[rows, :] = Datatypes[i] - A[rows, :]
        if (np.linalg.norm(X_est - (J + A))**2. < ConvergeThresh):
            break
        X_est = J + A
        if (j == MaxIter):
            print 'Warning: MaxIter iterations reached before convergence'
    # Transform back to original space
    J_original = []
    A_original = []
    for i in xrange(nDatatypes):
        ind1 = int(np.sum(d[:i]))
        ind2 = int(np.sum(d[:i+1]))
        Joint = J[ind1 : ind2, :]
        Indiv = A[ind1 : ind2, :]
        if (U_original[i] is not None):
            Joint = np.dot(U_original[i], Joint)
            Indiv = np.dot(U_original[i], Indiv)
        J_original.append(Joint.copy())
        A_original.append(Indiv.copy())
    J_original = np.vstack(J_original)
    A_original = np.vstack(A_original)
    #J_original = reshape(J_original, [-1, n])
    #A_original = reshape(A_original, [-1, n])
    return J_original, A_original

#################### JIVE_RankSelect.m #########################################

def JIVE_RankSelect(Datatypes, alpha, Nperm, scale=True, convThresh=1e-8, MaxIter=1000):
    ### WARNING: do copy an input sata to prevent corruption !!!!!!!!!!!!! (copy.deepcopy())
    '''
    %[J,A,r,rIndiv] = JIVE_RankSelect(Datatypes,alpha,Nperm,scale, convThresh)
    %Input:
    %    - Datatypes: Cell array of data matrices with matching columns (Datatypes{i} gives the i'th data matrix)
    %   - alpha: Significance threshold
    %   - Nperm: Number of permutations
    %   - scale: Should the datasets be centered and scaled by total variation? (default = 'y')
    %   - convThresh:  Convergence threshold to use for JIVE algorithm (default = 10^(-8))
    %   - MaxIter: Maximum numeber of iterations (default = 1000)
    %Output:
    %   -J,A: JIVE estimates
    %   - r: Rank of joint structure
    %   - rIndiv: Vector of ranks for individual structure.
    '''
    assert isinstance(Datatypes, (tuple, list)), 'Error: Datatypes should be either list or tuple'
    nDatatypes = len(Datatypes)
    d = [None]*nDatatypes
    n = [None]*nDatatypes
    for i in xrange(nDatatypes):
        d[i], n[i] = Datatypes[i].shape
    n = set(n)
    n = list(n)
    assert len(n) == 1, 'Error: Datatypes do not have same number of columns'
    n = n[0]
    if scale:
        for i in xrange(nDatatypes):
            Datatypes[i] -= np.mean(Datatypes[i], axis=1, keepdims=True)
            Datatypes[i] /= np.linalg.norm(Datatypes[i])
    # Estimate ranks
    A = np.zeros([np.sum(d), n])
    J = np.zeros([np.sum(d), n])
    rPrev = 1
    rIndivPrev = np.ones(nDatatypes)
    r = 0
    rIndiv = np.zeros(nDatatypes)
    First = True
    Resid = copy.deepcopy(Datatypes)
    SingValsI = [None]*nDatatypes
    SingValsPermI = [None]*nDatatypes
    PermThreshI = [None]*nDatatypes
    ResidPerm = [None]*nDatatypes
    while (r != rPrev) or (not np.all(rIndivPrev == rIndiv)):
        rPrev = r
        rIndivPrev = rIndiv
        U, S, Vt = fast_svd(J, rPrev)
        for k in xrange(nDatatypes):
            ind1 = int(np.sum(d[:k]))
            ind2 = int(np.sum(d[:k+1]))
            Resid[k] = Datatypes[k] - J[ind1 : ind2, :]
            prRes = np.dot(np.dot(Resid[k], Vt.T), Vt)
            _, SingValsI[k], _ = fast_svd(Resid[k] - prRes)
            SingValsPermI[k] = np.zeros([Nperm, len(SingValsI[k])])
            for i in xrange(Nperm):
                ResidPerm[k] = Resid[k].copy()
                for j in xrange(d[k]):
                    randPermInd = np.random.permutation(n)
                    ResidPerm[k][j, :] = ResidPerm[k][j, randPermInd]
                prRes = np.dot(np.dot(Resid[k], Vt.T), Vt)
                _, SingValsPermI[k][i, :], _ = fast_svd(ResidPerm[k] - prRes)

            PermThreshI[k] = np.percentile(SingValsPermI[k], (1-alpha)*100,
                        interpolation='midpoint', axis=0)
            rIndiv[k] = 0
            for i in xrange(min(d[k], n)):
                if (SingValsI[k][i] > PermThreshI[k][i]):
                    rIndiv[k] = i
                else:
                    break
        _, SingVals, _ = fast_svd(np.vstack(Datatypes) - A)
        SingValsPerm = np.zeros(Nperm, min(sum(d), n))
        for i in xrange(Nperm):
            for k in xrange(nDatatypes):
                Resid[k] = Datatypes[k] - A[sum(d[:k]) : sum(d[:k+1]), : ]
                randPermInd = np.random.permutation(n)
                ResidPerm[k] = Resid[k][:, randPermInd]
            _, SingValsPerm[i, :], _ = fast_svd(np.vstack(ResidPerm))
        PermThresh = np.percentile(SingValsPerm, (1-alpha)*100,
                        interpolation='midpoint', axis=0)
        r = 0
        for i in xrange(min(sum(d), n)):
            if SingVals[i] > PermThresh[i]:
                r = i+1
            else:
                break
        ## only increase rank of joint structure (generally this results in more accurate rank estimation).  
        r = max(r, rPrev) 
        print 'rank Individual :',  rIndiv
        J, A = JIVE(Datatypes, r, rIndiv, scale, convThresh, MaxIter)
    return J, A, r, rIndiv
