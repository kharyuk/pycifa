import numpy as np
import time
try:
    import mkl
    # global
    _MAX_NUM_OF_THREADS = mkl.get_max_threads()
    mkl.set_num_treads(_MAX_NUM_OF_THREADS)
except:
    pass
from utils import mldivide
from utils import fast_svd

def sobi(Y, p=4):#, maxEncore=1000):
    '''
     program  by  A. Belouchrani and A. Cichocki
     Second Order Blind Identification (SOBI)
    **************************************************
     blind identification by joint diagonalization   *
     of correlation  matrices.                       *
                           			  *
     ------------------------------------------------*
     THIS CODE ASSUMES TEMPORALLY CORRELATED SIGNALS *
     in estimating the cumulants                     *
     ------------------------------------------------*

     [H,S]=SOBI(X,m,p) produces a matrix H of dimension [m by n] and a matrix S
     of dimension [n by N] wich are respectively an estimate of the mixture matrix and
     an estimate of the source signals of the linear model from where the
     observation matrix X of dimension [m by N] derive.
     Note: > m: sensor number.
           > n: source number by default m=n.
           > N: sample number.
        > p: number of correlation matrices to be diagonalized by default p=4.

     REFERENCES:
     A. Belouchrani, K. Abed-Meraim, J.-F. Cardoso, and E. Moulines, ``Second-order
      blind separation of temporally correlated sources,'' in Proc. Int. Conf. on
      Digital Sig. Proc., (Cyprus), pp. 346--351, 1993.

     A. Belouchrani and K. Abed-Meraim, ``Separation aveugle au second ordre de
      sources correlees,'' in  Proc. Gretsi, (Juan-les-pins),
      pp. 309--312, 1993.

      A. Belouchrani, and A. Cichocki,
      Robust whitening procedure in blind source separation context,
      Electronics Letters, Vol. 36, No. 24, 2000, pp. 2050-2053.

      A. Cichocki and S. Amari,
      Adaptive Blind Signal and Image Processing, Wiley,  2002.

     Improve to process with a time delay vetor p
     p = 4:            M: m x m x 4
     p = [ 4 5]        M: m x m x 2    p = 4 and p = 5
     Code by Phan Anh Huy 28022007
    '''
    eps = 1e-10
    X = Y.copy()
    m, N = X.shape
    if isinstance(p, int):
        p = np.arange(p)
        
    pm = len(p)*m # for convenience
    X = X - np.mean(X, axis=1, keepdims=True)
    # prewhitening
    UU, S, VVt = np.linalg.svd(X.T, full_matrices=False)
    Q = (VVt.T / S).T
    iQ = VVt.T * S
    X = np.dot(Q, X)
    ### correlation matrices estimation
    k = 0
    lenS = S.size
    M = np.zeros([m, pm], dtype=np.complex)
    for u in xrange(0, pm, m):
        Rxp = np.dot(X[:, p[k]:N], X[:, : N-p[k]].T) / (N - p[k]) # p[0]=0
        M[:, u : u+m] = np.linalg.norm(Rxp) * Rxp
        k += 1
    ### joint diagonalization
    epsil = 1./(N**0.5 * 100)
    encore = True
    V = np.eye(m, dtype=np.complex)
    numEncore = 0
    indG = int(np.ceil(float(pm-m+1)/m))
    g = np.zeros([3, indG], dtype=np.complex)
    while (encore):# and (numEncore < maxEncore):
        encore = False
        for p in xrange(m-1):
            for q in xrange(p, m):
                ### Givens rotations
                g[0, :] = M[p:p+1, p:pm:m] - M[q:q+1, q:pm:m]
                g[1, :] = M[p:p+1, q:pm:m] + M[q:q+1, p:pm:m]
                g[2, :] = 1j*(M[q:q+1, p:pm:m] - M[p:p+1, q:pm:m])
                tmp = np.dot(g, g.T)
                vcp, D, _ = np.linalg.svd(tmp.real)
                #D = D**2.
                #[la,K]=sort(diag(D));
                angles = vcp[:, 0] ## matlab - ascending, python - descending
                angles = np.sign(angles[0]) * angles
                c = (0.5*(1. + angles[0]))**0.5
                sr = 0.5*(angles[1] - 1j*angles[2])/c
                sc = sr.conjugate()
                oui = abs(sr) > epsil
                encore = (encore) or (oui)
                if oui:  ### update of the M and V matrices
                    colp = M[:, p:pm:m].copy()
                    colq = M[:, q:pm:m].copy()
                    M[:, p:pm:m] = c*colp + sr*colq
                    M[:, q:pm:m] = c*colq - sc*colp
                    rowp = M[p, :].copy()
                    rowq = M[q, :].copy()
                    M[p, :] = c*rowp + sc*rowq
                    M[q, :] = c*rowq - sr*rowp
                    temp = V[:, p].copy()
                    V[:, p] = c*V[:, p] + sr*V[:, q]
                    V[:, q] = c*V[:, q] - sc*temp
        #numEncore += encore
    ### estimation of the mixing matrix and source signals
    H = np.dot(iQ, V) # estimated mixing matrix
    S = np.dot(V.T, X) # estimated sources
    if (np.linalg.norm(H.imag) >= eps) or (np.linalg.norm(S.imag) >= eps) or\
      (np.linalg.norm(D.imag) >= eps):
        print "sobi(): non-zero imaginary part"
    return H.real, S.real, D.real

def PMFsobi(Y, c=None, p=4):
    '''
     call the sobi algorithm
     usage: S, w = PMFsobi(X, m, p)
      m: source number
      p: number of correlation matrices to be diagonalized by default p=4.
      X: X=S*A, where the columns of S are the sources, A is the mixing matrix.
      
      Based on the version developed by A. Belouchrani and A. Cichocki.
    '''
    # the devil is in the details
    # here was a root of evil
    m = c
    X = Y.copy()
    if m is None:
        m = X.shape[1]

    z = pca_psc(X, m).T

    A, _, _ = sobi(z, p)

    S = mldivide(A, z).T
    w = mldivide(S, X)
    return S, w


def pca_psc(x, n):
    ## return principal subspace of the columns of x
    #    x : m-by-T
    #    u : m-by-n
    M, T = x.shape
    if n == T:
        return x.copy()
    if M >= T:
        z = np.dot(x.T, x)
        v, d, _ = np.linalg.svd(z)
        lenD = d.size
        u = np.dot(x, v[:, :lenD] / (d**0.5)) 
    else:
        z = np.dot(x, x.T)
        u, d, _ = np.linalg.svd(z)
    return u[:, :n]
