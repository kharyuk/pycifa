import time
import numpy as np
import matplotlib.pyplot as plt

import sklearn.cluster
import sklearn.neighbors

try:
    import pycifa
except:
    import sys
    sys.path.append('../')

from pycifa import constructW
from pycifa import MutualInfo
from pycifa import GNMF
from pycifa import tsne
from pycifa import cobe
from pycifa import mmc_nonnegative
from pycifa import accuracy

from pycifa.utils import princomp
from pycifa.utils import NormalizeFea
from pycifa.tools import loadmat


_DIRNAME_DATA = '../data/'

# TODO: verbose for inner functions
def demo_faceClustering(
        db='YALE',
        dirname=_DIRNAME_DATA,
        mcRun=5,
        cN=2,
        verbose=False
):
    if db.lower() == 'ORL'.lower():
        ## ORL
        filename = 'ORL_32x32.mat'
        nList = [30, 40] ## randomly selected clusters
    elif db.lower() == 'PIE'.lower():
        ## PIE
        filename = 'PIE_pose27.mat'
        #nList = [30, 40, 68]
        nList = [30]
    elif db.lower() == 'YALE'.lower():
        ## YALE
        filename = 'YaleB_32x32.mat'
        nList = range(11, 16)
    else:
        raise ValueError('db must be "ORL", "PIE", or "YALE"')
    df = loadmat(dirname + filename)
    fea = df['fea']
    gnd = df['gnd']
    if issubclass(fea.dtype.type, np.unsignedinteger) or (fea.max() > 100):
    #if strcmpi(class(fea),'uint8')||max(fea(:))>100
        fea = fea.astype(np.double)
        fea /= fea.max()
    nCol = min(np.sum(gnd==0), 200) # number of columns in A{n}
    nCol = max(nCol, 20)
    ## basic information
    NC = max(gnd) # number of classes
    Ln = len(nList)
    N = len(gnd) / nCol # number of groups. A{1,...,N}
    idx = 0
    ac = np.zeros([len(nList), mcRun, 5])
    mu = np.zeros([len(nList), mcRun, 5])
    for n in nList:
        for run in xrange(mcRun):
            if verbose:
                print 'n=%d  run=%d ...\n' % (n, run+1)
            order = np.random.permutation(NC)
            order = order[:n].copy()
            A = [None]*n
            gndn = []
            ## get data
            for i in xrange(n):
                ic = order[i]
                flag = gnd==ic
                A[i] = (fea[flag, :].copy()).T
                gndn += [i]*sum(flag)
            ## randperm
            os = np.random.permutation(len(gndn))
            A = np.hstack(A)[:, os]
            gndn = np.array(gndn)[os]
            g = int(np.floor((len(gndn)-1) / nCol))
            if (len(gndn) - g*nCol) < nCol:
                g -= 1
            gps = np.zeros(g+len(gndn))
            gps[:g] = nCol
            gps[g:] = len(gndn) - g*nCol
            gps = np.cumsum(gps)
            gps = gps[gps < A.shape[1]]
            ## generating data
            A = np.split(A, gps, axis=1)    ############################33  
            ## CIFA+tSNE
            c, Q, _ = cobe(A, c=cN)
            ## ifeature
            ifea = np.hstack(A) - np.dot(c, np.hstack(Q))
            #[nc nQ]=cnfe(c,Q);
            #ifea=[A{:}]-nc*cell2mat(nQ);
            ifea = tsne(ifea.T, None, 2, verb=False)
            
            kmeans = sklearn.cluster.KMeans(n_clusters=n, init='k-means++',
                n_init=20, max_iter=300, tol=0.0001, precompute_distances='auto',
                verbose=0, random_state=None, copy_x=True, n_jobs=1
            )
            kmeans.fit(ifea)
            le = kmeans.predict(ifea)
            ac[idx, run, 0] = accuracy(gndn, le)
            mu[idx, run, 0] = MutualInfo(gndn, le)
            ## tSNE
            ifea = tsne(np.hstack(A).T, None, 2, verb=False)
            kmeans.n_init = 50
            kmeans.fit(ifea)
            le = kmeans.predict(ifea)            
            #le, _, _ = kmeans(ifea, n, 'replicate', 50)
            ac[idx, run, 1] = accuracy(gndn, le)
            mu[idx, run, 1] = MutualInfo(gndn, le)
            ## PCA
            #ifea=compute_mapping([A{:}]','PCA',50);
            ifea, Lmat, _ = princomp(np.hstack(A), wtype='econ')
            ifea = ifea[:, :50]
            kmeans.n_init = 20
            kmeans.fit(ifea)
            le = kmeans.predict(ifea)   
            #le, _, _ = kmeans(ifea, n, 'replicate', 20)
            ac[idx, run, 2] = accuracy(gndn, le)
            mu[idx, run, 2] = MutualInfo(gndn, le)
            ## GNMF 
            gfea = NormalizeFea(np.hstack(A).T)
            gopts = {'WeightMode': 'Binary', 'maxIter': 100, 'alpha': 100}
            W = constructW(gfea, WeightMode=gopts['WeightMode'])#,
                #maxIter=gopts['maxIter'],
                # alpha=gopts['alpha'])
            W = np.array(W.todense())
            U, V, _, _ = GNMF(gfea.T, n, W, #WeightMode=gopts['WeightMode'])#,
                maxIter=gopts['maxIter'], alpha=gopts['alpha'])
            np.random.seed(5489) #######################
            kmeans.n_init = 20
            kmeans.fit(ifea)
            glb = kmeans.predict(ifea) 
            #glb, _, _ = kmeans(V, n, 'replicate', 20)    
            ac[idx, run, 3] = accuracy(gndn, glb)
            mu[idx, run, 3] = MutualInfo(gndn, glb)
            ## MMC
            Qini = np.random.rand(W.shape[0], n)
            Qmc, _, _, _ = mmc_nonnegative(W, Qini)
            u = np.max(Qmc, axis=1)
            mle = np.argmax(Qmc, axis=1)
            ac[idx, run, 4] = accuracy(gndn, mle)
            mu[idx, run, 4] = MutualInfo(gndn, mle)
        idx += 1
    toc = time.clock()
    # info='RUN 50 from demo_face. nc=2 pca=50. order=cobe-pca-tsne-gnmf-mmc tsne=2';
    # save PIE27_120507_a5.mat  ac mu info nList RUN
    algstr = 'COBE    tSNE      PCA      GNMF      MMC'
    if len(nList) == 1:
        mmu = np.squeeze(np.mean(mu, axis=1))
        mac = np.squeeze(np.mean(ac, axis=1))
    else:
        mmu = np.squeeze(np.mean(mu, axis=1))
        mac = np.squeeze(np.mean(ac, axis=1))
    print '    ==== Averaged Accuracy ===='
    print algstr
    print mac
    print '\n'
    print '    ==== Averaged MutualInfo ===='
    print algstr
    print mmu
