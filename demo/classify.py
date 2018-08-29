import numpy as np
from scipy.spatial.distance import correlation
import matplotlib.pyplot as plt

import sklearn.discriminant_analysis
import sklearn.neighbors

try:
    import pycifa
except:
    import sys
    sys.path.append('../')

from pycifa import cobe_classify

from pycifa.utils import vec

from pycifa.tools import loadmat

_DIRNAME_DATA = '../data/'

def demo_classify(
        filename='YaleB_32x32.mat',
        dirname=_DIRNAME_DATA,
        mcRun=50,
        minTrainRatio=0.2,
        maxTrainRatio=0.6,
        stepTrainRatio=0.1,
        verbose=False,
        datfname='cobe_classify',
        imfname='cobe_classify.pdf'
):
    cobe_opts = {}
    df = loadmat(dirname + filename)
    fea = df['fea'].T
    gnd = df['gnd']
    #if (issubclass(fea.dtype.type, np.unsignedinteger)) or (fea.max() > 100):
    #    fea /= fea.max()
    #Ti = (gnd==1).sum()
    gnd -= gnd.min()
    CLS = set(gnd)
    CLS = list(CLS)
    train_ratio = np.arange(minTrainRatio, maxTrainRatio+stepTrainRatio,
        stepTrainRatio)
    train_ratio = train_ratio[train_ratio <= maxTrainRatio+1e-1*stepTrainRatio]
    idx = 0
    T = fea.shape[1]
    ac = np.zeros([train_ratio.size, mcRun, 3])
    for r in train_ratio:
        for run in xrange(mcRun):
            if verbose:
                print 'r=%.5f  run=%d/%d' % (r, run+1, mcRun)
            # Randomly generate training set and test set
            trFea = []
            trLs = []
            teFea = []
            teLs = []
            # here was a mistake with cidx (in the end, cidx += 1)
            for cidx in xrange(len(CLS)):
                c = CLS[cidx]
                Ti = np.sum(gnd==c)
                trTi = int(np.floor(Ti*r))
                feac = fea[:, gnd==c]
                teTi = np.sum(gnd==c) - trTi
                o = np.random.permutation(Ti)
                trFea.append( feac[:, o[ :trTi]].copy() )
                trLs.append( np.tile(cidx, (1, trFea[cidx].shape[1])) )
                teFea.append( feac[:, o[trTi:]] .copy() )
                teLs.append( np.tile(cidx, (1, teFea[cidx].shape[1])) )
            ## A -- training
            A = np.hstack(trFea).T
            teFea = np.hstack(teFea).T
            teLs = np.hstack(teLs)
            #teLs = vec(teLs).tolist()
            trLs = np.hstack(trLs)
            trLs = vec(trLs).tolist()
            ## Running classification algorithms
            # teFea: test data
            #     A: training data
            #  trLs: labels of training data
            ## cobe based classifier
            algidx = 0
            cobe_opts['subgroups'] = max(2, np.floor(trTi/200))
            cobe_opts['cobeAlg'] = 'cobe'
            p = cobe_classify(teFea, A, trLs, subgroups=cobe_opts['subgroups'],
                cobeAlg=cobe_opts['cobeAlg']) ##############
            ac[idx, run, algidx] = (np.sum(teLs == p) / float(teLs.size))*100.
            ## KNN
            algidx = 1
            u, d, vt = np.linalg.svd(A)
            nSV = min(50, d.size)
            vt = vt[:nSV, :]
            le = knnclassify(np.dot(teFea, vt.T), np.dot(A, vt.T), trLs, 5,
                             'correlation')
            ac[idx, run, algidx] = (np.sum(teLs == le) / float(teLs.size))*100.
            ## LDA classify diaglinear
            algidx = 2
            u, d, vt = np.linalg.svd(A)
            nSV = min(50, d.size)
            vt = vt[:nSV, :]
            le = classify(np.dot(teFea, vt.T), np.dot(A, vt.T), trLs, 'linear')
            ac[idx, run, algidx] = (np.sum(teLs == le) / float(teLs.size))*100.
        idx += 1
    mac = np.squeeze(np.mean(ac, axis=1))
    stds = np.squeeze(np.std(ac, axis=1, ddof=1))

    if datfname is not None:
        np.savez_compressed(datfname, ac=ac, mac=mac, std=stds)

    cs = [[ 10, 36, 106],
          [216, 41,   0],
          [  0,  0, 255]]
    cs = np.array(cs)/255.
    lbls = ['COBEc','KNN','LDA']
    plt.clf()
    for idx in xrange(3):
        plt.errorbar(train_ratio, mac[:, idx], yerr=stds[:, idx],
                    ecolor=cs[idx, :], color=cs[idx, :], linewidth=2, label=lbls[idx])
    plt.title('Classification of Yale-B')
    plt.axis('tight')
    plt.grid(True)
    #set(gca,'XTick',train_ratio,'XTickLabel',train_ratio); #############################
    plt.xticks(train_ratio, train_ratio)
    plt.ylim([30, 100])
    plt.xlim([0.15, 0.65])
    plt.legend(loc='upper left')
    plt.xlabel('Size of training data (%)')
    plt.ylabel('Accuracy (%)')
    if imfname:
        plt.savefig(imfname)
    else:
        plt.show()
    
def knnclassify(Sample, Training, Group, k, distance):
    #custdist = lambda x, y: 1. - np.corrcoef(x, y, rowvar=False, ddof=1)[0, 1]
    metric = {
        'euclidean': 'euclidean',
        'cityblock': ' manhattan',
        'cosine': 'haversine',
        'correlation': 'correlation',
        'hamming': 'hamming'
    }
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, 
        metric=metric[distance], algorithm='brute'
    )
    neigh.fit(Training, Group)
    labels = neigh.predict(Sample)
    return labels
    
def classify(sample, training, group, ctype, prior=None):
    if ctype == 'linear':
        clfr = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd',
            shrinkage=None, priors=prior, n_components=None, store_covariance=False,
            tol=0.0001
        )
    elif ctype == 'quadratic':
        clfr = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(
            priors=prior, reg_param=0.0, store_covariances=False, tol=0.0001
        )
    else:
        raise NotImplementedError()
    clfr.fit(training, group)
    labels = clfr.predict(sample)
    return labels
    

