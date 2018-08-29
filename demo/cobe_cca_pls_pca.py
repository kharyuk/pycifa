import numpy as np
import matplotlib.pyplot as plt

import sklearn.cross_decomposition

try:
    import pycifa
except:
    import sys
    sys.path.append('../')

from pycifa import JIVE
from pycifa import cobe

from pycifa.utils import datanormalize
from pycifa.utils import mldivide

from pycifa.tools import addStringInFilename


_DIRNAME_DATA = '../data/'


def demo_cobe_cca_pls_pca(
        rC=1,
        rI=9,
        T=1000,
        datfname='demo_cobe_cca_pls_pca.npz',
        imfname='demo_cobe_cca_pls_pca.pdf'
):
    M = rI + rC
    s = np.zeros([T, 2])
    t = np.arange(0, T)
    s[:, 0] = np.sin(1e-2*t)
    s[:, 1] = np.sign(s[:, 0])
    A = []
    tmp = np.zeros([T, rI+1])
    tmp[:, 0] += s[:, 0]
    tmp[:, 1:rI+1] += np.random.randn(T, rI)
    tmp = np.dot(tmp, np.random.randn(rI+1, rI+1))
    A.append(tmp.copy())
    tmp = np.zeros([T, rI+1])
    tmp[:, 0] += s[:, 1]
    tmp[:, 1:rI+1] += np.random.randn(T, rI)
    tmp = np.dot(tmp, np.random.randn(rI+1, rI+1))
    A.append(tmp.copy())
    
    c, _, _ = cobe(A, rC)
    tmp = mldivide(A[0], c)
    px = np.dot(A[0], tmp)
    tmp = mldivide(A[1], c)
    py = np.dot(A[1], tmp)
    
    plsReg = sklearn.cross_decomposition.PLSRegression(n_components=rC,
        scale=True, max_iter=500, tol=1e-06, copy=True
    )
    plsReg.fit(A[0], A[1])
    xl = plsReg.x_loadings_
    yl = plsReg.y_loadings_
    xs = plsReg.x_scores_
    ys = plsReg.y_scores_
    xpls, _ = datanormalize(xs[:, :1]) ############
    ypls, _ = datanormalize(ys[:, :1]) ############
    
    wa, wb, r, xl, yl = canoncorr(A[0], A[1])
    xcca, _ = datanormalize(np.dot(A[0], wa[:, :rC]))
    ycca, _ = datanormalize(np.dot(A[1], wb[:, :rC]))
    
    Amat = np.hstack(A)
    pca, d, v = np.linalg.svd(Amat)
    nSV = min(d.size, rC)
    pca = pca[:, :nSV].copy()
    
    At = [(x.copy()).T for x in A]
    JAc, JAi = JIVE(At, r=rC, rIndiv=[rI, rI], scale=True, ConvergeThresh=1e-10,
        MaxIter=5000
    )
    temp, d, JAc = np.linalg.svd(JAc)
    nSV = min(rC, d.size)
    JAc = JAc[:nSV, :].T

    fig, ax = plt.subplots(2, 1, figsize=[8, 5])
    #figure('units','normalized','outerposition',[0.2 0.55 0.6 0.4],'Name','COBE vs. CCA');
    ax[0].plot(px, label=r'$\mathbf{Y}_1\mathbf{w}_1$')
    ax[0].plot(py, label=r'$\mathbf{Y}_2\mathbf{w}_2$')
    ax[0].plot(c, label=r'$\mathbf{\bar{a}}_1$')
    ax[0].axis('tight')
    ax[0].set_xlabel('COBE')
    ax[0].legend()
    ax[1].plot(xcca, label=r'$\mathbf{Y}_1\mathbf{w}_1$')
    ax[1].plot(ycca, label=r'$\mathbf{Y}_2\mathbf{w}_2$')
    ax[1].axis('tight')
    ax[1].set_xlabel('Canonical Correlation Analysis');
    ax[1].legend()
    fig.tight_layout()
    if imfname is not None:
        tmp = addStringInFilename(imfname, '_cca', prefix=False)
        plt.savefig(tmp)
    else:
        plt.show()

    plt.clf()
    #figure('units','normalized','outerposition',[0.2 0.3 0.6 0.25],'Name','COBE vs. PLS');
    fig = plt.figure()
    sig = np.sign(np.corrcoef(xpls, c, rowvar=False)[0,1])
    plt.plot(xpls, label='PLS-$\mathbf{Y}_1$')
    plt.plot(ypls, label='PLS-$\mathbf{Y}_2$')
    plt.plot(sig*c, label=r'$\mathbf{\bar{a}}_1$')
    plt.legend()
    plt.xlabel('Partial Least Squares');
    plt.axis('tight')
    if imfname is not None:
        tmp = addStringInFilename(imfname, '_pls', prefix=False)
        plt.savefig(tmp)
    else:
        plt.show()

    #figure('units','normalized','outerposition',[0.2 0.05 0.6 0.25],'Name',);
    plt.clf()
    plt.figure()
    sig = np.sign(np.corrcoef(pca, c, rowvar=False)[0,1])
    sigJ = np.sign(np.corrcoef(pca, JAc, rowvar=False)[0, 1])
    plt.plot(pca, label='PCA')
    plt.plot(sigJ*JAc, label='JIVE')
    plt.plot(sig*c, label='COBE')
    #ls=get(gca,'child');
    #set(ls(1),'color','red','linewidth',2);
    plt.legend()
    plt.title('COBE vs. PCA and JIVE')
    plt.axis('tight')
    if imfname is not None:
        tmp = addStringInFilename(imfname, '_battle', prefix=False)
        plt.savefig(tmp)
    else:
        plt.show()
        
def canoncorr(X, Y, rank=None):
    '''
    # https://github.com/clab/clab-scalable-cca/blob/master/cca_qr.m
    # note: function assumes that X and Y are un-normalized (i.e., not scaled
    # by number of examples like 1/(n-1))
    # QR not a good idea when X is tall - decomposition is expensive and
    # destroys sparsity - Q is dense and very large!
    '''
    n = X.shape[0]
    q_x, r_x = np.linalg.qr(X)
    q_y, r_y = np.linalg.qr(Y)
    U, S, Vt = np.linalg.svd(np.dot(q_x.T, q_y))
    nSV = S.size

    U_correct = U[:, :nSV] * np.sqrt(n-1)
    V_correct = Vt[:nSV, :].T * np.sqrt(n-1) 
    A = mldivide(r_x, U_correct)
    B = mldivide(r_y, V_correct)

    u = np.dot(q_x, U_correct)
    v = np.dot(q_y, V_correct)
    return A, B, S, u, v
