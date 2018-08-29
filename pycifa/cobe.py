import numpy as np
import copy
from utils import mldivide
from cnfe import cnfe


def cobe(Y_in, c=None, maxiter=2000, PCAdim=0.8, tol=1e-6, epsilon=3e-2, returnBc=True):
    # Common orthogonal basis extraction
    Y = copy.deepcopy(Y_in)
    Ydim = [x.shape[0] for x in Y]
    NRows = Y[0].shape[0]
    N = len(Ydim)
    assert np.all(np.array(Ydim) == NRows), 'Y must have the same number of rows.'
    U = []
    J = np.zeros(N, dtype='i')
    for n in xrange(N):
        tmp, _ = np.linalg.qr(Y[n])
        U.append(tmp)
        J[n] = U[n].shape[1]
    ## some matrices are of full-row rank. Pre-processing by using PCA is required.
    flag = J > NRows
    if np.sum(flag) > 1:
        if PCAdim < 1:
            rdims = np.max(np.floor(NRows * PCAdim), 2)
        else:
            rdims = np.min(NRows - 1, PCAdim)
        for n in xrange(N):
            if flag[n]:
                tmp, d, vt = np.linalg.svd(Y[n])
                U[n] = tmp[:, :rdims]
                J[n] = rdims
    minJn = int(np.min(J))
    x = []
    for n in xrange(N):
        x.append(np.zeros([U[n].shape[1], minJn]))
    ## Seeking the first common basis
    Ac = np.dot(U[0], np.random.randn(J[0], 1))
    Ac = Ac / np.linalg.norm(Ac)
    for it in xrange(maxiter):
        c0 = Ac.copy()
        c1 = np.zeros([NRows, 1])
        for n in xrange(N):
            tmp = np.dot(U[n].T, Ac)
            x[n][:, :1] = tmp
            c1 += np.dot(U[n], x[n][:, :1])
        Ac = c1 / np.linalg.norm(c1)
        if np.abs((c0*Ac).sum()) > 1-tol:
            break
    res = []
    res.append(0)
    for n in xrange(N):
        tmp = np.dot(U[n].T, Ac)
        res[0] += 1 - (tmp*tmp).sum()
    res[0] /= float(N)
    if (res[0] > epsilon) and (c is None):  ## c is not specified. 
        print '%s' % ('No common basis found.')
        Ac = None
        Bc = copy.deepcopy(Y)
        # Zi = copy.deepcopy(Y)
        return Ac, Bc, res
    if c is not None:
        c = int(np.min([c, J.min()]))
        res = res + [np.inf]*(c-1)
        Ac = np.hstack([Ac, np.zeros([NRows, c-1])])
    else:
        res = res + [np.inf]*(minJn-1)
        Ac = np.hstack([Ac, np.zeros([NRows, minJn-1])])
    ## seeking the residual common basis
    for j in xrange(1, minJn):
        ## ## stopping criteria -- 1 where c is given
        if (c is not None) and (j >= c):
            break
        ## update U;
        for n in xrange(N):
            tmp = np.dot(U[n], x[n][:, j-1:j]) ## j-1?
            U[n] -= np.dot(tmp, x[n][:, j-1:j].T)
        Ac[:, j:j+1] = np.dot(U[0], np.random.randn(U[0].shape[1], 1))
        Ac[:, j:j+1] /= np.linalg.norm(Ac[:, j:j+1])
        ## get another column
        for it in xrange(maxiter):
            c0 = Ac[:, j:j+1].copy()
            c1 = np.zeros([NRows, 1])
            for n in xrange(N):
                x[n][:, j:j+1] = np.dot(U[n].T, Ac[:, j:j+1])
                c1 += np.dot(U[n], x[n][:, j:j+1])
            Ac[:, j:j+1] = c1 / np.linalg.norm(c1)
            if abs((c0*Ac[:, j:j+1]).sum()) > 1-tol:
                break
        res[j] = 0
        for n in xrange(N):
            tmp = np.dot(U[n].T, Ac[:, j:j+1])
            res[j] += 1 - (tmp*tmp).sum()
        res[j] /= N
        ## stopping criteria -- 2
        if (res[j] > epsilon) and (c is None):
            res[j] = np.inf
            break
    res = np.array(res)
    Ac = Ac[:, res != np.inf]
    res = res[res != np.inf]
    if returnBc:
        Bc = []
        for n in xrange(N):
            tmp = np.dot(Ac.T, Y[n])
            Bc.append(tmp)
        return Ac, Bc, res
    return Ac, None, res

def cobec(Y_in, c=1, maxiter=200, ctol=1e-3, PCAdim=0.8, retBZ=True):
    '''
    %COMSSPACE Summary of this function goes here
    %   Detailed explanation goes here
    '''
    Y = copy.deepcopy(Y_in)
    Ydim = [x.shape[0] for x in Y]
    NRows = Y[0].shape[0]
    N = len(Ydim)
    assert np.all(np.array(Ydim) == NRows), 'Wrong dimension of Y.'
    U = [None]*N
    x = [None]*N
    Ac = np.zeros([Y[0].shape[0], c])
    J = np.zeros(N)
    for n in xrange(N):
        U[n], _ = np.linalg.qr(Y[n])   
        J[n] = U[n].shape[1]
        assert J[n] >= c, 'Rank deficient / c is too large.'
        if J[n] > NRows:
            if PCAdim < 1:
                J[n] = max(int(np.floor(NRows*PCAdim)), 2)
            else:
                J[n] = min(NRows-1, int(np.floor(PCAdim)))
            U[n], d, vt = np.linalg.svd(Y[n])
            nSV = min(d.size, J[n])
            U[n] = U[n][:, :nSV].copy()
        ## initialize
        x[n] = np.random.randn(U[n].shape[1], c)
        x[n], _ = np.linalg.qr(x[n])
        Ac += np.dot(U[n], x[n])
        u, temp, vt = np.linalg.svd(Ac)
        Ac = np.dot(u[:, :c], vt[:c, :])
    ## iterations
    x = [None]*N
    for it in xrange(maxiter):
        c0 = Ac.copy()
        c2 = np.zeros([NRows, c])
        for n in xrange(N):
            x[n] = np.dot(U[n].T, Ac)
            c2 += np.dot(U[n], x[n])
        #[u , temp, v]=svds(c2,c,'L');
        #Ac=u*v';
        Ac, _ = np.linalg.qr(c2)   
        ## stop
        tmp1 = np.dot(Ac.T, c0)
        tmp1 = np.diag(tmp1)
        if (it > 20) and (np.mean(np.abs(tmp1)) > 1-ctol):
            break
    if retBZ:
        Bc = [None]*N
        Zi = [None]*N
        for n in xrange(N):
            Bc[n] = np.dot(Ac.T, Y[n])
            Zi[n] = mldivide(Y[n], Ac)
        return Ac, Bc, Zi
    return Ac

def pcobe(Y_in, c=None, maxiter=2000, PCAdim=0.8, tol=1e-6, epsilon=0.03, pdim=None,
    returnBcZi=False):
    # mldivide()
    # Common orthogonal basis extraction
    # Usage: 
    Y = copy.deepcopy(Y_in)
    Ydim = [x.shape[0] for x in Y]
    NRows = Y[0].shape[0]
    N = len(Ydim)
    assert np.all(np.array(Ydim) == NRows), 'Y must have the same number of rows.'
    if pdim is None:
        pdim = max([x.shape[1] for x in A]) # what is "A"?
        pdim = min(NRows, pdim)
    P = np.random.randn(pdim, NRows)
    U = []
    J = np.zeros(N, dtype='i')
    PY = []
    for n in xrange(N):
        tmp = np.dot(P, Y[n])
        PY.append(tmp.copy())
        tmp, _ = np.linalg.qr(PY[n])    
        U.append(tmp.copy())
        J[n] = U[n].shape[1]
    ## some matrices are of full-row rank.  Pre-processing by using PCA is required.
    flag = J > pdim
    if np.sum(flag) > 1:
        if PCAdim < 1:
            rdims = np.max(np.floor(NRows * PCAdim), 2)
        else:
            rdims = np.min(pdim, PCAdim)
        for n in xrange(N):
            if flag[n]:
                tmp, d, vt = np.linalg.svd(PY[n])
                U[n] = tmp[:, :rdims]
                J[n] = rdims
    minJn = int(np.min(J))
    x = []
    for n in xrange(N):
        x.append(np.zeros([U[n].shape[1], minJn]))
    ## Seeking the first common basis
    Ac = np.dot(U[0], np.random.randn(J[0], 1))
    Ac = Ac / np.linalg.norm(Ac)
    for it in xrange(maxiter):
        c0 = Ac.copy()
        c1 = np.zeros([pdim, 1])
        for n in xrange(N):
            tmp = np.dot(U[n].T, Ac)
            x[n][:, :1] = tmp
            c1 += np.dot(U[n], x[n][:, :1])
        Ac = c1 / np.linalg.norm(c1)
        if np.abs((c0*Ac).sum()) > 1-tol:
            break
    res = []
    res.append(0)
    for n in xrange(N):
        tmp = np.dot(U[n].T, Ac)
        res[0] += 1 - (tmp*tmp).sum()
    res[0] /= float(N)
    if (res[0] > epsilon) and (c is None):  ## c is not specified. 
        print '%s' % ('No common basis found.')
        Ac = None
        Bc = copy.deepcopy(Y)
        # Yi = copy.deepcopy(Y)
        return Ac, Bc, res

    if c is not None:
        c = int(np.min([c, J.min()]))
        res = res + [np.inf]*(c-1)
        Ac = np.hstack([Ac, np.zeros([pdim, c-1])])
    else:
        res = res + [np.inf]*(minJn-1)
        Ac = np.hstack([Ac, np.zeros([pdim, minJn-1])])
    ## seeking the residual common basis
    for j in xrange(1, minJn):
        ## ## stopping criteria -- 1 where c is given
        if (c is not None) and (j >= c):
            break
        ## update U
        for n in xrange(N):
            tmp = np.dot(U[n], x[n][:, j-1:j]) ## j-1?
            U[n] -= np.dot(tmp, x[n][:, j-1:j].T)
        Ac[:, j:j+1] = np.dot(U[0], np.random.randn(U[0].shape[1], 1))
        Ac[:, j:j+1] /= np.linalg.norm(Ac[:, j:j+1])
        ## get another column
        for it in xrange(maxiter):
            c0 = Ac[:, j:j+1].copy()
            c1 = np.zeros([pdim, 1])
            for n in xrange(N):
                x[n][:, j:j+1] = np.dot(U[n].T, Ac[:, j:j+1])
                c1 += np.dot(U[n], x[n][:, j:j+1])
            Ac[:, j:j+1] = c1 / np.linalg.norm(c1)
            if abs((c0*Ac[:, j:j+1]).sum()) > 1-tol:
                break
        res[j] = 0
        for n in xrange(N):
            tmp = np.dot(U[n].T, Ac[:, j:j+1])
            res[j] += 1 - (tmp*tmp).sum()
        res[j] /= N
        ## stopping criteria -- 2
        if (res[j] > epsilon) and (c is None):
            res[j] = np.inf
            break
    res = np.array(res)
    Ac = Ac[:, res != np.inf]
    res = res[res != np.inf]
    ## proj
    cP = np.zeros([NRows, c])
    for n in xrange(N):
        x[n] = mldivide(PY[n], Ac)
        cP += np.dot(Y[n], x[n])
    Ac, _ = np.linalg.qr(cP)
    if returnBcZi:
        Zi = []
        Bc = []
        for n in xrange(N):
            Bc.append(np.dot(Ac.T, Y[n]))
            Zi.append(mldivide(Y[n], Ac))
        return Ac, Bc, res, Zi
    Bc=None
    return Ac, Bc, res

def cobe_classify(testData, training, group, nc=None, subgroups=2, nn=False,
                  dist='correlation', cobeAlg='pcobe', pdim=None):
    '''
    %COBE_CLASSIFY Summary of this function goes here
    %   Detailed explanation goes here
    %dist: correlation|Euclid
    %cobe: cobe|cobec|pcobe
    '''
    eps = np.spacing(1)
    glabs = set(group)
    glabs = list(glabs)
    nfea = training.shape[1]
    ## extract feature
    Ac = [None]*len(glabs)
    for idx in xrange(len(glabs)):
        flag = np.array(group) == glabs[idx]
        A = training[flag, :]
        ## split
        if subgroups > 1:
            trT = np.sum(flag)
            nCol = int(np.floor(trT / subgroups))
            n0 = int(np.floor(trT / nCol))
            if (trT - n0*nCol) < nCol:
                n0 -= 1
            split = [nCol]*n0 + [trT - n0*nCol]
            split = np.cumsum(split)
            split = split[split < A.shape[0]]
            #if min(split) > A.shape[0]:
                #raise ValueError('Error specification of the number of subgroups.')
            if nc is None:
                cobe_opts_c = np.floor(min(split)*0.8)
            else:
                cobe_opts_c = c
            cobealg = cobeAlg.lower()
            tmp_in = np.split(A[:, :nfea].T, split, axis=1)
            if cobealg == 'pcobe':
                if pdim is None:
                    pdim = np.ceil(A.shape[1] * 0.5)
                else:
                    pdim = min(pdim, A.shape[1])
                cobe_opts_pdim = 1000 # why is not pdim? why 1000?
                tmp_out, Q, _ = pcobe(tmp_in, c=cobe_opts_c, pdim=cobe_opts_pdim) ########3
            elif cobealg == 'cobec':
                tmp_out, Q, _ = cobec(tmp_in, c=cobe_opts_c, retBZ=True)
            elif cobealg == 'cobe':
                tmp_out, Q, _ = cobe(tmp_in, c=cobe_opts_c)
            else:
                raise NotImplementedError('Unsupported algorithm.')
            if nn:
                Ac[idx] = cnfe(tmp_out, Q)
            else:
                Ac[idx] = tmp_out.copy()
        else: 
            Ac[idx] = (A.copy()).T
    ## classifying
    teT = testData.shape[0]
    labels = np.zeros(teT)
    if dist == 'correlation':
        dis = np.zeros([len(glabs), teT])
        ## testdata normalization
        testData = testData.T
        testData = testData - np.mean(testData, axis=0, keepdims=True) # cast same_rule restriction
        tmp = max(np.linalg.norm(testData), eps)
        testData /= tmp
        ## template normalization
        for idx in xrange(len(glabs)):
            Ac[idx] -= np.mean(Ac[idx], axis=0, keepdims=True)
            Ac[idx], _ = np.linalg.qr(Ac[idx])
            proj = np.dot(Ac[idx].T, testData)
            dis[idx, :] = np.sum(proj**2., axis=0)**0.5
        labels = np.argmax(dis, axis=0)  ########################### 
        labels = np.array(glabs)[labels]
    else:
        raise NotImplementedError('Unsupported distance.')
    return labels
