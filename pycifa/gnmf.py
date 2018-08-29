import numpy as np
import scipy.sparse
import copy


def GNMF(X_in, k, W_in, U_in=None, V_in=None, error=1e-5, maxIter=None, nRepeat=10, minIter=30,
meanFitRatio=0.1, alpha=100, alpha_nSmp=False, optimization='Multiplicative', weight=None):
    '''
    % Graph regularized Non-negative Matrix Factorization (GNMF)
    %
    % Notation:
    % X ... (mFea x nSmp) data matrix 
    %       mFea  ... number of words (vocabulary size)
    %       nSmp  ... number of documents
    % k ... number of hidden factors
    % W ... weight matrix of the affinity graph 
    %
    % options ... Structure holding all settings
    %               options.alpha ... the regularization parameter. 
    %                                 [default: 100]
    %                                 alpha = 0, GNMF boils down to the ordinary NMF. 
    %                                 
    %
    % You only need to provide the above four inputs.
    %
    % X = U*V'
    %
    % References:
    % [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
    % Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
    % (ICDM'08), Pisa, Italy, Dec. 2008. 
    %
    % [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
    % Non-negative Matrix Factorization for Data Representation", IEEE
    % Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
    % 8, pp. 1548-1560, 2011.  
    %
    %
    %   version 2.0 --April/2009 
    %   version 1.0 --April/2008 
    %
    %   Written by Deng Cai (dengcai AT gmail.com)
    '''
    X = X_in.copy()
    W = W_in.copy()
    U = copy.deepcopy(U_in)
    V = copy.deepcopy(V_in)
    assert X.min() >= 0, 'Input should be nonnegative!'
    nSmp = X.shape[1]
    if alpha_nSmp:
        alpha *= nSmp
    if weight is not None:
        if weight == 'NCW':
            feaSum = np.sum(X, axis=1)
            D_half = np.dot(X.T, feaSum)
            X = np.dot(X, 
            np.array(scipy.sparse.spdiags(D_half**-0.5, 0, nSmp, nSmp).todense()))
    if (optimization.lower() == 'Multiplicative'.lower()):
        U_final, V_final, nIter_final, objhistory_final = GNMF_Multi(X, k, W, U, V,
            differror=error, minIter=minIter, meanFitRatio=meanFitRatio,
            maxIter=maxIter, nRepeat=nRepeat, alpha=alpha)
    else:
        raise NotImplementedError
    return U_final, V_final, nIter_final, objhistory_final

#################### GNMF_Multi.m ##############################################

def GNMF_Multi(X, k, W, U=None, V=None, differror=1, minIter=10, meanFitRatio=1, nRepeat=1,
                alpha=0, NormW=False, Converge=False, maxIter=None): ##############################################################3
    '''
    % Graph regularized Non-negative Matrix Factorization (GNMF) with
    %          multiplicative update
    % Notation:
    % X ... (mFea x nSmp) data matrix 
    %       mFea  ... number of words (vocabulary size)
    %       nSmp  ... number of documents
    % k ... number of hidden factors
    % W ... weight matrix of the affinity graph 
    %
    % options ... Structure holding all settings
    %
    % You only need to provide the above four inputs.
    %
    % X = U*V'
    %
    % References:
    % [1] Deng Cai, Xiaofei He, Xiaoyun Wu, and Jiawei Han. "Non-negative
    % Matrix Factorization on Manifold", Proc. 2008 Int. Conf. on Data Mining
    % (ICDM'08), Pisa, Italy, Dec. 2008. 
    %
    % [2] Deng Cai, Xiaofei He, Jiawei Han, Thomas Huang. "Graph Regularized
    % Non-negative Matrix Factorization for Data Representation", IEEE
    % Transactions on Pattern Analysis and Machine Intelligence, , Vol. 33, No.
    % 8, pp. 1548-1560, 2011.  
    %
    %
    %   version 2.1 --Dec./2011 
    %   version 2.0 --April/2009 
    %   version 1.0 --April/2008 
    %
    %   Written by Deng Cai (dengcai AT gmail.com)
    '''
    minIter -= 1
    if (maxIter is not None) and (maxIter < minIter):
        minIter = maxIter
    Norm = 2
    NormV = 0
    mFea, nSmp = X.shape
    objhistory = []
    objhistory_final = []
    if alpha > 0:
        W = alpha*W
        DCol = np.sum(W, axis=1)
        D = scipy.sparse.spdiags(DCol.flatten(), 0, nSmp, nSmp).todense()
        D = np.array(D)
        L = D - W
        L = np.array(L)
        if NormW:
            D_mhalf = scipy.sparse.spdiags(DCol**-0.5, 0, nSmp, nSmp).todense()
            D_mhalf = np.array(D_mhalf)
            L = np.dot(np.dot(D_mhalf, L), D_mhalf)
    else:
        L = None
    selectInit = True
    if (U is None) or (V is None):
        U = np.abs(np.random.rand(mFea, k))
        V = np.abs(np.random.rand(nSmp, k))
    else:
        nRepeat = 1 ##################
    U, V = NormalizeUV(U, V, NormV, Norm)
    if (nRepeat == 1):
        selectInit = False
        minIter = 0
        if maxIter is None:
            tmp, _ = CalculateObj(X, U, V, L)
            objhistory = [tmp]
            meanFit = list(np.array(objhistory)*10)
        else:
            if Converge:
                tmp, _ = CalculateObj(X, U, V, L)
                objhistory = [tmp]
    else:
        if Converge:
            raise NotImplementedError('Not implemented!')
    tryNo = 0 #################
    nIter = 0 ##################
    while tryNo < nRepeat: ############
        tryNo += 1
        maxErr = 1
        while (maxErr > differror):
            # ===================== update V ========================
            XU = np.dot(X.T, U) # mnk or pk (p<<mn)
            UU = np.dot(U.T, U) # mk^2
            VUU = np.dot(V, UU) # nk^2
            if alpha > 0:
                WV = np.array(W.dot(V))
                DV = np.dot(D, V)
                XU += WV
                VUU += DV
            VUU[VUU < 1e-10] = 1e-10        
            V = V * (XU / VUU)
            # ===================== update U ========================
            XV = np.dot(X, V) # mnk or pk (p<<mn)
            VV = np.dot(V.T, V) # nk^2
            UVV = np.dot(U, VV) # mk^2
            UVV[UVV < 1e-10] = 1e-10
            tmp = (XV / UVV) # 3mk
            U *= tmp
            nIter += 1
            if nIter > minIter:
                if selectInit:
                    tmp, _ = CalculateObj(X, U, V, L)
                    objhistory = [tmp]
                    maxErr = 0
                else:
                    if maxIter is None:
                        newobj, _ = CalculateObj(X, U, V, L)
                        objhistory = objhistory + [newobj] ##ok<AGROW>
                        meanFit = meanFitRatio * meanFit + (1 - meanFitRatio)*newobj
                        maxErr = (meanFit - newobj) / meanFit
                    else:
                        if Converge:
                            newobj, _ = CalculateObj(X, U, V, L)
                            objhistory = objhistory + [newobj] ##ok<AGROW>
                        maxErr = 1
                        if (nIter >= maxIter):
                            maxErr = 0
                            if Converge:
                                pass
                            else:
                                objhistory.append(0)
        if (tryNo == 1): ###################
            U_final = U.copy()
            V_final = V.copy()
            nIter_final = nIter
            objhistory_final = copy.deepcopy(objhistory)
        else:
           if objhistory[-1] < objhistory_final[-1]:
               U_final = U.copy()
               V_final = V.copy()
               nIter_final = nIter
               objhistory_final = copy.deepcopy(objhistory)
        if selectInit:
            if (tryNo < nRepeat): #######################33
                # re-start
                U = np.abs(np.random.rand(mFea, k))
                V = np.abs(np.random.rand(nSmp, k))
                U, V = NormalizeUV(U, V, NormV, Norm)
                nIter = 0
            else:
                tryNo -= 1
                nIter = minIter + 1
                selectInit = False
                U = U_final.copy()
                V = V_final.copy()
                objhistory = copy.deepcopy(objhistory_final)
                meanFit = list(np.array(objhistory)*10)
    U_final, V_final = NormalizeUV(U_final, V_final, NormV, Norm)
    return U_final, V_final, nIter_final, objhistory_final

def CalculateObj(X, U, V, L, deltaVU=False, dVordU=True):
    # 500M. You can modify this number based on your machine's computational power.
    MAXARRAY = 500*1024*1024/8 
    dV = []
    nSmp = X.shape[1]
    mn = len(X)
    nBlock = np.ceil(mn/MAXARRAY)
    if (mn < MAXARRAY):
        dX = np.dot(U, V.T) - X
        obj_NMF = np.sum(dX**2.)
        if deltaVU:
            if dVordU:
                dV = np.dot(dX.T, U) + np.dot(L, V)
            else:
                dV = np.dot(dX, V)
    else:
        obj_NMF = 0
        if deltaVU:
            if dVordU:
                dV = np.zeros(V.shape)
            else:
                dV = np.zeros(U.shape)
        PatchSize = np.ceil(nSmp/nBlock)
        for i in xrange(nBlock):
            if (i*PatchSize > nSmp):
                smpIdx = tuple(range(i*PatchSize+1, nSmp))
            else:
                smpIdx = tuple(range(i*PatchSize+1, (i+1)*PatchSize))
            dX = np.dot(U, V[smpIdx, :].T - X[:, smpIdx])
            obj_NMF += np.sum(dX**2.)
            if deltaVU:
                if dVordU:
                    dV[smpIdx, :] = np.dot(dX.T, U)
                else:
                    dV = dU + np.dot(dX, V[smpIdx, :])
        if deltaVU:
            if dVordU:
                dV += np.dot(L, V)
    if L is None:
        obj_Lap = 0
    else:
        obj_Lap = np.sum(np.dot(V.T, L)*V.T)
    obj = obj_NMF + obj_Lap
    return obj, dV

def NormalizeUV(U, V, NormV, Norm):
    K = U.shape[1]
    if Norm == 2:
        if NormV:
            norms = np.sqrt(np.sum(V**2., axis=0))
            norms[norms < 1e-15] = 1e-15
            V = np.dot(V, scipy.sparse.spdiags(1./norms, 0, K, K).todense())
            U = np.dot(U, scipy.sparse.spdiags(norms, 0, K, K).todense())
        else:
            norms = np.sqrt(np.sum(U**2., axis=0))
            norms[norms < 1e-15] = 1e-15
            U = np.dot(U, scipy.sparse.spdiags(1./norms, 0, K, K).todense())
            V = np.dot(V, scipy.sparse.spdiags(norms, 0, K, K).todense())
    else:
        if NormV:
            norms = np.sqrt(np.sum(np.abs(V), axis=0))
            norms[norms < 1e-15] = 1e-15
            V = np.dot(V, scipy.sparse.spdiags(1./norms, 0, K, K).todense())
            U = np.dot(U, scipy.sparse.spdiags(norms, 0, K, K).todense())
        else:
            norms = np.sqrt(np.sum(np.abs(U), axis=0))
            norms[norms < 1e-15] = 1e-15
            U = np.dot(U, scipy.sparse.spdiags(1./norms, 0, K, K).todense())
            V = np.dot(V, scipy.sparse.spdiags(norms, 0, K, K).todense())
    U = np.array(U)
    V = np.array(V)
    return U, V
