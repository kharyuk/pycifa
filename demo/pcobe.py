import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import pycifa
except:
    import sys
    sys.path.append('../')

from pycifa import pcobe
from pycifa import CalcSIR
from pycifa import PMFsobi

from pycifa.utils import addGaussianNoise

from pycifa.tools import loadmat


_DIRNAME_DATA = '../data/'


def demo_pcobe(
        filename='Speech4.mat',
        dirname=_DIRNAME_DATA,
        mcRun=50,
        noiseL=20,
        N=10,
        c=4,
        K=6,
        Jn=50,
        pdimsMin=100,
        pdimsMax=1000,
        pdimsStep=100,
        verbose=False,
        datfname='demo_pcobe',
        imfname='demo_pcobe.pdf'
):
    '''
    # mcRun : number of Monte-Carlo runs
    # noiseL : level of additive noise;
    # N : number of matrices
    # c : number of common components. 
    # K : number of individual components
    # Jn : number of Yn's columns
    '''
    df = loadmat(dirname + filename)
    Ac = df['Speech4'].T
    T, nColsAc = Ac.shape
    pdims = np.arange(pdimsMin, pdimsMax+pdimsStep, pdimsStep)
    pdims = pdims[pdims <= pdimsMax+1e-1*pdimsStep]
    sirs = np.zeros((pdims.size, mcRun))
    tims = np.zeros(pdims.size)
    bss_opts = {}
    bss_opts['NumOfComp'] = c
    rI = [K]*N
    tol = 1e-6
    maxiter = 1000
    for run in xrange(mcRun):
        if verbose:
            print 'P-COBE test: Run [%d/%d] ...' % (run+1, mcRun)
        ## Re-generating observations
        Y = []
        for n in xrange(N):
            tmp = np.zeros([T, c+K])
            tmp[:, :nColsAc] += Ac.copy()
            tmp[:, c:c+K] = np.random.randn(T, K)
            tmp = np.dot(tmp, np.random.randn(c+K, Jn))
            tmp = addGaussianNoise(tmp, noiseL)
            Y.append(tmp.copy())
        ## pCOBE
        for pidx in xrange(pdims.size):
            pdim = pdims[pidx]
            if verbose:
                print 'dim=%d  run=[%d/%d]' % (pdim, run+1, mcRun)
            ts = time.clock()
            eBc, _, _ = pcobe(Y, c=c, maxiter=maxiter, tol=tol,
                            pdim=pdim, returnBcZi=False)
            se, ae = PMFsobi(eBc[:, :c], c=bss_opts['NumOfComp'])
            #jh
            tims[pidx] += (time.clock() - ts)
            #print se.conj().sum()
            # TODO: warning (casting complex values to real-valued array)
            tmp = CalcSIR(Ac, se)
            sirs[pidx, run] = np.mean(tmp)
            '''
            plt.clf()
            a = Ac.copy()# / np.linalg.norm(Ac, axis=0, keepdims=True)
            a -= np.mean(a, axis=0, keepdims=True)
            #a = whiten(a)
            a /= np.std(a, axis=0, keepdims=True, ddof=1)
            a /= np.linalg.norm(a, axis=0, keepdims=True)
            b = se.copy() #/ np.linalg.norm(se, axis=0, keepdims=True)
            b -= np.mean(b, axis=0, keepdims=True)
            b /= np.std(b, axis=0, keepdims=True, ddof=1)
            b /= np.linalg.norm(b, axis=0, keepdims=True)
            print (a**2).sum(axis=0), ((a-b)**2.).sum(axis=0)
            plt.plot(a, 'r')
            plt.plot(b, 'g')
            plt.show()'''
        # save result
        if datfname is not None:
            np.savez_compressed(datfname, sirs=sirs, mcRun=mcRun, tims=tims)
    tims /= mcRun
    msirs = np.mean(sirs, axis=1) ## axis mean(,2)
    print ' ==== Results ===='
    print 'Reduced dims:       ', pdims, '  / Total = ', T
    print 'Averaged SIRs (dB): ', msirs
    print 'Averaged Time  (s): ', tims
    ## plot
    fig, ax1 = plt.subplots()
    ax1.set_title('Performance of PCOBE')
    ax1.plot(pdims, tims, 'r>-', marker='>', markerfacecolor='r', markeredgecolor='r')
    ax1.set_xlim([50, 1050])
    ax1.set_ylim([min(tims), max(tims)])
    ax1.set_ylabel('Averaged running time (s)', color='r')
    ax1.tick_params('y', colors='r')
    ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(pdims, msirs, 'bs-', marker='s', markerfacecolor='b', markeredgecolor='b')
    ax2.set_xlim([50, 1050])
    ax2.set_ylabel('Averaged SIR (dB)', color='b')
    ax2.set_xlabel('$\itI_p$,  ($\itI$=5000, $\itJ_n$=50, $\itc$=4.)')
    ax2.tick_params('y', colors='b')
    fig.tight_layout()
    if imfname is not None:
        plt.savefig(imfname)
    else:
        plt.show()
