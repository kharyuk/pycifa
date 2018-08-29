# cython: boundscheck=False, wraparound=False, nonecheck=False
#import cython
#cimport cython
import numpy as np
#cimport numpy as np

# only for integer p
def sobi(X, pIn=4, maxEncore=100):
    cdef double eps = 1e-10
    cdef int m, n
    m, N = X.shape
    cdef int pm = pIn*m # for convenience
    X = X - np.mean(X, axis=0, keepdims=True)
    # prewhitening
    UU, S, VVt = np.linalg.svd(X.T, full_matrices=False)
    cdef int nS = S.size
    VVt = VVt[:nS, :]
    Q = (VVt.T / S).T
    iQ = VVt.T * S
    X = np.dot(Q, X)
    ### correlation matrices estimation
    cdef int k = 0
    cdef int u, p, q
    cdef complex sc
    cdef double sr, c, ts
    cdef int oui
    ip = np.arange(pIn)
    M = np.zeros([nS, pm], dtype=np.complex)
    for u in xrange(0, pm, m):
        Rxp = np.dot(X[:, ip[k]:N], X[:, : N-ip[k]].T) / (N - ip[k]) # p[0]=0
        M[:, u : u+m] = np.linalg.norm(Rxp) * Rxp
        k += 1
    ### joint diagonalization
    cdef double epsil = 1./(N**0.5 * 100)
    cdef int encore = 1
    V = np.eye(m, dtype=np.complex)
    cdef int numEncore = 0
    cdef int indG = int(np.ceil(float(pm-m+1)/m))
    g = np.zeros([3, indG], dtype=np.complex)
    while (encore == 1) and (numEncore < maxEncore):
        encore = 0
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
                if oui == 1:  ### update of the M and V matrices
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
        numEncore += encore
    ### estimation of the mixing matrix and source signals
    H = np.dot(iQ, V) # estimated mixing matrix
    S = np.dot(V.T, X) # estimated sources

    return H.real, S.real, D.real

