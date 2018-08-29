import numpy as np
import matplotlib.pyplot as plt

try:
    import pycifa
except:
    import sys
    sys.path.append('../')

from pycifa import cobe
from pycifa import cnfe

from pycifa.utils import reshape

from pycifa.tools import loadmat

_DIRNAME_DATA = '../data/'

def demo_heart(
        filename='heart.mat',
        dirname=_DIRNAME_DATA,
        N=4,
        rC=2,
        rI=8,
        nMix=20,
        origShape=[164, 164],
        datfname='demo_heart',
        imfname='demo_heart.pdf'
):
    ## demo for simulation 2
    # load two gray-scaled x-ray images with shape (164 x 164)
    # stored as vectors
    if not dirname.endswith('/'):
        dirname += '/'
    df = loadmat(dirname + filename)
    S = df['S'].T
    nRows, nCols = S.shape
    A = []
    for n in xrange(N):
        tmp = np.zeros([nRows, rC+rI])
        tmp[:, :rC] = S[:, :rC].copy()
        tmp[:, rC : rC+rI] = np.random.rand(nRows, rI)
        ## mixing
        tmp = np.dot(tmp, np.random.rand(rC+rI, nMix))
        A.append(tmp)
    c, Q, _ = cobe(A, c=rC)  # common features extraction
    nc, nQ = cnfe(c, Q, r=rC) # nonnegative common features extraction
    # save result
    if datfname is not None:
        np.savez_compressed(datfname, obs=A, rec=nc)
    ## visualization of rersults.
    plt.clf()
    f, ax = plt.subplots(3, N)#, figsize = [9, 12])
    for k in xrange(N):
        ax[0, k].imshow(reshape(A[k][:, 0], origShape), interpolation='none', cmap=plt.cm.gray)
        ax[0, k].axis('off')
    ax[0, 0].set_title('Four observations')
    for k in xrange(rC):
        ax[1, k].imshow(reshape(nc[:, k], origShape), interpolation='none', cmap=plt.cm.gray)
        ax[1, k].axis('off')
    for k in xrange(rC, N):
        ax[1, k].axis('off')
    ax[1, 0].set_title('Recovered signals')
    for k in xrange(nCols):
        ax[2, k].imshow(reshape(S[:, k], origShape), interpolation='none', cmap=plt.cm.gray)
        ax[2, k].axis('off')
    for k in xrange(nCols, N):
        ax[2, k].axis('off')
    ax[2, 0].set_title('Original signals')
    if imfname is not None:
        plt.savefig(imfname)
    else:
        plt.show()
