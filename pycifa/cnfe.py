import numpy as np
from utils import mldivide


def cnfe(c, Q, r=None, maxiter=2000, tol=1e-6, alpha=0, ncheck=10):
# This function will extract common nonnegative features by solving
#  min \sum_n ||c*Q{n}-nc*nQ||
#   s.t. nc>=0  nQ>=0
# Usage: [ nc nQ ] = cnfe( c, Q,opts );
# 
    NMFALG = 'mu'
    eps = np.spacing(1)
    M, rc = c.shape
    if (r is None) or (r > rc):
        r = rc
    nc = np.random.rand(M, r)
    N = len(Q)
    nQ = []
    for n in xrange(N):
        tmp = np.dot(mldivide(nc, c), Q[n])
        tmp[tmp < eps] = 1e-3
        nQ.append(tmp)
    if NMFALG == 'mu':
        for it in xrange(maxiter):
            ## update nc first
            nc0 = nc.copy()
            num = np.zeros([rc, r])
            denominator = np.zeros([r, r])
            for n in xrange(N):
                num += np.dot(Q[n], nQ[n].T)
                denominator += np.dot(nQ[n], nQ[n].T)
            for it2 in xrange(5):             # what is happening here? why 5? why "it", not another variable? (in orig. file)?
                tmp1 = np.dot(c, num)
                tmp1[tmp1 < eps] = eps
                tmp2 = np.dot(nc, denominator)
                tmp2[tmp2 < eps] = eps
                nc *= tmp1 / tmp2
            #nc = bsxfun(@rdivide,nc,max(sum(nc),eps));
            ## update nQ{n} for each n
            nctc = np.dot(nc.T, c)
            nctnc = np.dot(nc.T, nc)
            for n in xrange(N):
                for it2 in xrange(5): # what is happening here? why 5? why "it", not another variable? (in orig. file)?
                    tmp1 = np.dot(nctc, Q[n])
                    tmp1[tmp1 < eps] = eps
                    tmp2 = np.dot(nctnc, nQ[n]) + alpha*nQ[n]
                    tmp2[tmp2 < eps] = eps
                    nQ[n] *= tmp1 / tmp2
            ## stopping criterion
            if (it > maxiter*0.25) and (it % ncheck == 0):
                if np.linalg.norm(nc - nc0) < tol:
                    break
    else:
        raise NotImplemented('Unsupported algorithm.')
    return nc, nQ
