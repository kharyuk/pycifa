import time
import numpy as np
import matplotlib.pyplot as plt

import sklearn.cross_decomposition

try:
    import pycifa
except:
    import sys
    sys.path.append('../')
try:
    import mkl
    # global
    _MAX_NUM_OF_THREADS = mkl.get_max_threads()
    mkl.set_num_treads(_MAX_NUM_OF_THREADS)
except:
    pass

from pycifa import JIVE
from pycifa import PMFsobi
from pycifa import cobe
from pycifa import cobec
from pycifa import call_mcca
from pycifa import CalcSIR

from pycifa.utils import princomp
from pycifa.utils import addGaussianNoise

from pycifa.tools import loadmat

_DIRNAME_DATA = '../data/'

def demo_LinkedBSS(filename='Speech4.mat', dirname=_DIRNAME_DATA, mcRun=50, c=4,
N=10, K=6, Jn=50, noiseL=20, cobe_opts=None, cobec_opts=None,
bss_opts=None):
    ALG = 5
    '''
    mcRun - Monte-Carlo Runs
    c - Number of common components
    N - Number of matrices
    K - Number of individual components
    Jn - Number of columns of Yn
    noiseL - Level of additive noise
    '''
    df = loadmat(dirname + filename)
    Ac = df['Speech4']
    T = Ac.shape[1]
    sirs = np.zeros([mcRun, c, ALG])
    tims = np.zeros(ALG)
    rI = [K]*N
    if cobe_opts is None:
        cobe_opts = {}
        cobe_opts['epsilon'] = 0.03
        cobe_opts['maxiter'] = 1000
    if cobec_opts is None:
        cobec_opts = {}
        cobec_opts['c'] = c
        cobec_opts['maxiter'] = 1000
    if bss_opts is None:
        bss_opts = {}
        bss_opts['c'] = c
    for run in xrange(mcRun):
        print 'Run [%d/%d] ...' % (run+1, mcRun)
        ## Re-generating observations
        Y0 = [None]*N
        Y = [None]*N
        for n in xrange(N):
            tmp = np.zeros([c+K, T])
            tmp[:c, :T] = Ac.copy()
            tmp[c : c+K, :] = np.random.randn(K, T)
            Y[n] = np.dot(np.random.randn(Jn, c+K), tmp)
            Y0[n] = (Y[n].copy()).T  ## for c-detection
            ##Y{n}=awgn(Y{n},noiseLevel,'measured');
            Y[n] = addGaussianNoise(Y[n], noiseL)
        print 'JIVE is running ...'
        #r, rIndiv = JIVE_RankSelect(Y, 0.05, 100) ###########
        #print 'selected r and rIndiv:', r, rIndiv
        algindex = 0
        ts = time.clock()
        J, X = JIVE(Y, c, rI, True, 1e-5, 10) ################
        w, h = PMFsobi(J.T, c=bss_opts['c'])
        tims[algindex] += time.clock() - ts
        sirs[run, :, algindex] = np.sort(CalcSIR(Ac.T, w)) ################################################################################################################
        # transpose Y
        Y = [x.T for x in Y]
        print 'COBE is running ...'
        ## COBE
        algindex = 1
        ts = time.clock()
        eBc, ex, _ = cobe(Y, epsilon=cobe_opts['epsilon'],
            maxiter=cobe_opts['maxiter']
        )
        if eBc is not None:
            se, ae = PMFsobi(eBc[:, :c], c=bss_opts['c'])
            tims[algindex] += time.clock() - ts
            sirs[run, :, algindex] = np.sort(CalcSIR(Ac.T, se))###################
        print 'COBEc is running ...'
        ## COBEc
        algindex = 2
        ts = time.clock()
        eBc, ex, _ = cobec(Y, c=cobec_opts['c'], maxiter=cobec_opts['maxiter'])
        se, ae = PMFsobi(eBc[:, :c], c=bss_opts['c'])
        tims[algindex] += time.clock() - ts
        sirs[run, :, algindex] = np.sort(CalcSIR(Ac.T, se)) #################    '''
        ## MCCA   ---- joint BSS
        print 'JBSS is running ...'
        algindex = 3
        ts = time.clock()
        Se, _ = call_mcca(Y, c)
        tims[algindex] += time.clock() - ts
        si = np.zeros(c)
        for n in xrange(N):
            si += CalcSIR(Ac.T, Se[n].real) ############################33
        print 'JBSS Without SOBI: ', np.sort(si / N)
        si = np.zeros(c)
        for n in xrange(N):
            tmp, _ = PMFsobi(Se[n])
            si += CalcSIR(Ac.T, tmp)
        sirs[run, :, algindex] = np.sort(CalcSIR(Ac.T, se)) #################
        print 'JBSS With SOBI: ', np.sort(si / N)
        ## PCA 
        algindex = 4
        ts = time.clock()
        coe, pcs, _ = princomp(np.hstack(Y), wtype='econ')
        pcs = pcs[:, :c]
        #[pcs d v]=svds([Y{:}],c,'L');
        se, ae = PMFsobi(eBc[:, :c], c=bss_opts['c'])
        tims[algindex] += time.clock() - ts
        sirs[run, :, algindex] = np.sort(CalcSIR(Ac.T, se)) #################   
    tims /= mcRun
    msir = np.squeeze(np.mean(sirs, axis=0))
    print '\n'
    print ' ============= RESULTS ============'
    print ' --  Mean SIRs (dB) -- '
    print 'JIVE    COBE    COBEc   JBSS    PCA'
    print msir
    print '---------------------'
    print np.mean(msir)
    print ' \n \n'
    print '                        JIVE    COBE     COBEc   JBSS    PCA'
    print '  Time consumption (s):  ', tims
    # save demo_ica.mat sirs tims
    ######### EXP II
    ## detection of number of components
    ## single test
    cobe_opts['c'] = 20 #######################################
    estAc, Q, res = cobe(Y, c=cobe_opts['c'], maxiter=cobe_opts['maxiter'])
    plt.figure()
    #hfig=figure('units','inch','position',[1 1 4.5 3],'visible','off');
    idx = np.arange(1, 5)
    c1 = np.array([10, 36, 106]) / 255
    c2 = np.array([216, 41, 0]) / 255
    plt.plot(idx+1, res[idx], '.-', color=c1)
    idx = np.arange(4, cobe_opts['c'])
    plt.plot(idx+1, res[idx], 'x-', color=c2)
    plt.legend(['Common', 'Individual'], loc='lower right')
    plt.grid(True)
    plt.xlabel('Index \iti');
    plt.ylabel(r'$\frac{1}{N}f_i$')
    #movegui(hfig,'center'); #######################################3
    #set(get(gca,'child'),'linewidth',1,'MarkerSize',8); ######################
    #set(hfig,'visible','on');
    plt.show()

    ###### Full test
    #cobe_opts.c=20;
    #res=[];
    #noiseLevel = [20, 10, 0]
    #linestyle={'-','-.',':'};
    #for i in xrange(len(noiseLevel)):
        #print 'NoiseLevel: %d dB.\n' % (noiseLevel[i])
        #for n in xrange(N):
        #######Y[n] = awgn(Y0{n},noiseLevel(i),'measured');
            #Y[n] = addGaussianNoise(Y0[n], noiseLevel[i]) 
            #estAc, Q, res[i, :] = cobe(Y, cobe_opts) #################
    #figure('Name','fi');
    #c1 = np.array([10, 36, 106]) / 255
    #c2 = np.array([216, 41, 0]) / 255
    #hax1=axes; #########################
    #pos=get(hax1,'position'); ################################
    #for i in xrange(len(noiseLevel)):
        #for idx in xrange(4):
            #drawfunc(hax1,idx,res(i,idx),horzcat('s',linestyle{i}),'Color',c1,'MarkerFaceColor','none','MarkerEdgeColor',c1); #####################################
    #plt.xlim([1, cobe_opts['c']])
    #plt.ylim([0, 0.9])
    #hlgd1 = plt.legend(['20dB', '10dB', '0dB'])
    #hax2=axes; ########################
    #for i in xrange(len(noiseLevel)):
        #for idx in xrange(4, cobe_opts['c']):
            #drawfunc(hax2,idx,res(i,idx),horzcat('x',linestyle{i}),'Color',c2,'MarkerFaceColor',c2,'MarkerEdgeColor',c2);#############################################
    #plt.xlim([1, cobe_opts['c']])
    #plt.ylim([0, 0.9])
    #plt.grid(True)
    #hlgd2=legend({'20dB','10dB','0dB'},'Location','southeast'); ####################################
    #set(hax2,'Color','none','position',pos); ##########################3
    #plt.xlabel('Index \iti')
    #plt.ylabel('$\frac{1}{N}f_i$')
    #pl1=get(hlgd1,'position'); ####################3 
    #pl2=get(hlgd2,'position'); ########################
    #pl1=[pl2(1)-pl1(3)-0.005 pl2(2:4)]; ####################
    #set(hlgd1,'position',pl1); ################################



