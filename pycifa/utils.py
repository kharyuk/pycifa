import numpy as np

def reshape(x, shape):
    return np.reshape(x, shape, order='F')

def vec(x):
    return x.flatten(order = 'F')

def mldivide(a, b):
    assert a.ndim == b.ndim == 2
    m, n = a.shape
    if m == n:
        rv = np.linalg.solve(a, b)
    else:
        rv = np.linalg.lstsq(a, b)[0]
    return rv
    
def sub2ind(shape, ind):
    assert len(shape) == len(ind)
    if isinstance(ind, np.ndarray):
        assert 0 < ind.ndim < 3 
    else:
        tmp = [isinstance(x, int) for x in ind]
        assert np.all(tmp) == np.any(tmp) 
        if not np.all(tmp):
            tmp = [isinstance(x, (tuple, list, np.ndarray))  for x in ind]
            assert np.all(tmp) == np.any(tmp) 
            assert np.all([len(x) == len(ind[0]) for x in ind])
    indNP = np.array(ind)
    indNP = reshape(indNP, [-1, len(shape)])
    assert np.all(indNP.max(axis=0) < np.array(shape))
    N = np.cumprod(shape)
    N[1:] = N[:-1]
    N[0] = 1
    rind = np.dot(indNP, N)
    return tuple(rind)
    
def datanormalize(x, nor=2, ori=0):
    assert (isinstance(x, np.ndarray) and (x.ndim == 2)), "2-d numpy array is expected"
    xnorm = np.linalg.norm(x, nor, axis=ori)
    return x/xnorm, xnorm
    
def EuDist2(fea_a, fea_b=None, bSqrt=True):
    '''
    %EUDIST2 Efficiently Compute the Euclidean Distance Matrix by Exploring the
    %Matlab matrix operations.
    %   Written by Deng Cai (dengcai AT gmail.com)
    Python port
    '''
    if fea_b is None:
        aa = np.sum(fea_a*fea_a, axis=1, keepdims=True)
        ab = np.dot(fea_a, fea_a.T)
        D = aa + aa.T - 2*ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)
        D = np.maximum(D, D.T)
    else:
        aa = np.sum(fea_a*fea_a, axis=1, keepdims=True)
        bb = np.sum(fea_b*fea_b, axis=1, keepdims=True)
        ab = np.dot(fea_a, fea_b.T)
        D = aa + bb.T - 2*ab
        D[D < 0] = 0
        if bSqrt:
            D = np.sqrt(D)
    return D
    
def NormalizeFea(x, row=True):
    # if row == True: normalize each row of x to have unit norm
    # if row == False: normalize each column of x to have unit norm
    M, N = x.shape
    nrm = np.linalg.norm(x, axis=row, keepdims=True)
    return x / nrm
    
def imcomplement(A):
    if A.dtype == bool:
        B = not A
    elif issubclass(A.dtype.type, np.floating):
        B = 1. - A
    elif issubclass(A.dtype.type, np.integer):
        #if issubclass(A.dtype.type, np.signedinteger):
        B = ~A
    else:
        raise ValueError
    return B
    
def addGaussianNoise(s_in, SNR=20):
    ## This function is used to add i.i.d. Gaussian noise to the rows of s when
    ## 'awgn' is unavailable.
    s = s_in.copy()
    [R, T] = s.shape
    sn = np.zeros([R, T])
    powS = np.linalg.norm(s, axis=1, keepdims=True)
    noi = np.random.randn(R, T)
    powN = np.linalg.norm(noi, axis=1, keepdims=True)
    p = np.power(10, -SNR/20.) * powS/powN
    sn = s + p*noi
    return sn
    
def whiten(a):
    u, s, vt = np.linalg.svd(a)
    nS = s.size
    b = np.dot(u[:,:nS], vt[:nS, :])
    return b

def princomp(A, wtype='full'):
    """
     Matlab equivalet obtained via link:
     http://glowingpython.blogspot.ru/2011/07/principal-component-analysis-with-numpy.html
     and manually edited.
     performs principal components analysis 
         (PCA) on the n-by-p data matrix A
         Rows of A correspond to observations, columns to variables. 

     Returns :  
      coeff :
        is a p-by-p matrix, each column containing coefficients 
        for one principal component.
      score : 
        the principal component scores; that is, the representation 
        of A in the principal component space. Rows of SCORE 
        correspond to observations, columns to components.
      latent : 
        a vector containing the eigenvalues 
        of the covariance matrix of A.
    """
    n, p = A.shape
    M = (A - np.mean(A, axis=0, keepdims=True)).T
    fmf = wtype == 'econ'
    coeff, latent, vt = np.linalg.svd(M, full_matrices=fmf)
    if fmf:
        nSV = np.sum(latent != 0)
        if n <= p:
            nSV = min(nSV, n-1)
    else:
        nSV = latent.size
    score = (vt[:nSV, :].T * latent[:nSV]).T
    return coeff[:, :nSV], score, latent[:nSV]**2.
    
def fast_svd(x, maxRank=None, factor=1.5):
    eps = np.spacing(1)
    nRow, nCol = x.shape
    if float(nRow) / nCol > factor:
        z = np.dot(x.T, x)
        _, s, vt = np.linalg.svd(z)
        s = s**0.5
        nS = s[s>0].size
        s = s[:nS]
        if maxRank is not None:
            nS = min(nS, maxRank)
            s = s[:nS]
        vt = vt[:nS, :]
        u = np.dot(x, vt.T) / s
    elif float(nCol) / nRow > factor:
        z = np.dot(x, x.T)
        u, s, _ = np.linalg.svd(z)
        nS = s[s>0].size
        s = s[:nS]
        if maxRank is not None:
            nS = min(nS, maxRank)
            s = s[:nS]
        u = u[:, :nS]
        vt = (np.dot(u.T, x).T / s).T
    else:
        u, s, vt = np.linalg.svd(x)
        nS = s[s>0].size
        s = s[:nS]
        if maxRank is not None:
            nS = min(nS, maxRank)
        s = s[:nS]
        u = u[:, :nS]
        vt = vt[:nS, :]
    return u, s, vt
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

