import numpy as np
import copy

from utils import reshape


def call_mcca(X_all, numOfCV):
    ## call ssqcor_cca
    ## transpose the input
    X_all_dup = [X.T.copy() for X in X_all]#cellfun(@(x) x',X_all,'uni',false);
    ## pre-processing
    y = preprocess_mcca(X_all_dup)
    ## call MCCA
    B, theta_opt = ssqcor_cca_efficient(y, numOfCV)
    ## Output
    N = len(y)
    S = []
    for n in xrange(N):
        tmp = np.dot(B[n], y[n])
        S.append(tmp.T)
    return S, theta_opt

def preprocess_mcca(X_all):
## Pre-processing before joint-BSS: PCA + sphering
#  Extracted from exp_cca_multiset_bss_robustness_public.m
#   by G. Zhou
    K = len(X_all)
    numPCv = []
    B = []
    group_pc_org = []
    for i in xrange(K):
        mixedsig = X_all[i]
        numPCv.append(mixedsig.shape[0])
        ## PCA
        out = np.dot(mixedsig, mixedsig.T) / float(mixedsig.shape[1])
        V, S, _ = np.linalg.svd(out)
        S = S**2.
        lenS = S.size
        V = V[:, :lenS]
        data = np.dot(V.T, mixedsig)
        ## Sphering
        sphere_E = reshape(1./S[:numPCv[i]], [1, -1])
        data = (data[:numPCv[i], :].T * sphere_E).T
        ## Save PCA + sphering results
        B.append(V[:, :numPCv[i]] / sphere_E)##
        ## Pass the whitened data to 'group_pc_org' for M-CCA
        group_pc_org.append(data[:numPCv[i], :])
    return group_pc_org
    
def ssqcor_cca_efficient(y, numOfCV, B0=None, numMaxIter=1000, eps=1e-4):
    '''
    %% This code implement the M-CCA algorithm based on the SSQCOR cost 
    %% Reference:
    %% J. R. Kettenring, Canonical analysis of several sets of variables,?
    %% Biometrika, vol. 58, pp. 433?1, 1971.

    %% Input: 
    %% y: M by 1 cell array containing the *prewhitened* group datasets
    %% numOfCV: Number conanical vectors to be estimated
    %% B0: M by 1 cell array containing the initial guess of the demixing
    %% matrices for the group dataset: default is identity matrix
    %% Output:
    %% B: M by 1 cell array containing the estimated demixing matrices 
    %% theta_opt: Vector containing cost function values at the optimal
    %% solutions

    %% Yiou (Leo) Li Mar. 2009
    '''
    M = len(y)
    p, N = y[0].shape
    ## Calculate all pairwise correlation matrices
    R = [[None]*M]*M
    Rhat = [[None]*M]*M
    theta_opt = np.zeros(numOfCV)
    for i in xrange(M):
        for j in xrange(i, M):
            R[i][j] = np.dot(y[i], y[j].T) / N
            R[j][i] = (R[i][j]).T
        R[i][i] = np.eye(p)
    ## Obtain a prilimary estimate by MAXVAR algorithm
    if B0 is None:
        #B0 = maxminvar_cca(y, numOfCV);
        B0 = [None]*M
        for i in xrange(M):
            B0[i] = np.eye(numOfCV, p)
    B = copy.deepcopy(B0)
    for s in xrange(numOfCV):    
        ## Iterations to solve the s-th stage canonical vectors for 1--M
        ## datasets
        theta_old = np.zeros(M)
        theta = np.zeros(M)
        for n in xrange(numMaxIter):     
            if (n == 0):
                ## Initialize B{1--M}(s,:) by B0;
                for j in xrange(M):
                    B[j][s, :] /= np.linalg.norm(B[j][s, :])  ## Use normalized B0 (deepcopied)
                ## Calculate the cost funtion at the initial step
                #for j in xrange(M):
                    for k in xrange(M):
                        Rhat[j][k] = np.dot(B0[j][s, :], np.dot(R[j][k], B0[k][s, :].T))
                tmp = np.array(Rhat)
                theta_0 = np.trace(np.dot(tmp, tmp.T))
            ## Solve the current canonical vector for the j-th dataset
            for j in xrange(M):
                ## Calculate the cost function
                #jtheta_old[j] = 0
                #for k in xrange(M):
                #    jtheta_old[j] += np.dot(B[k][s, :], np.dot(R[k][j], B[j][s, :].T))
                ## Calculate the terms for updating jbn
                jC = B[j][:s, :].T #-1 or not?
                if (s > 0):
                    jA = np.eye(p) - np.dot(jC, jC.T) # *inv(jC'*jC)
                else:
                    jA = np.eye(p)
                jP = np.zeros([p, p])
                for k in xrange(M):
                    if (k != j):
                        tmp = np.dot(R[j][k], B[k][s, :].T)
                        jP += np.dot(tmp, tmp.T)
                ## update jbn
                z = np.dot(jA, jP)
                Ev, Dv, _ = np.linalg.svd(z)
                B[j][s, :] = Ev[:, 0].copy()
                tmp[j] = Dv[0] + 1 # should = jtheta(j)
                ## Calculate the cost function
                #jtheta[j] = 0
                #for k in xrange(M):
                #    jtheta[j] += np.dot(B[k][s, :], np.dot(R[k][j], B[j][s, :].T))
                #chec[j] = tmp[j] - jtheta[j]
                #delta[j] = jtheta[j] - jtheta_old[j]
            ## Calculate the cost funtion at the current step
            for j in xrange(M):
                for k in xrange(M):
                    Rhat[j][k] = np.dot(B[j][s, :], np.dot(R[j][k], B[k][s, :].T))
            tmp = np.array(Rhat)
            theta[n] = np.trace(np.dot(tmp, tmp.T))
            ## Check termination condition
            if (np.sum(abs(theta[n] - theta_old)) < eps) or (n == numMaxIter):
                theta_opt[s] = theta[n]
                break
            theta_old = theta[n]
    #print '\n Component # %d is estimated, in %d iterations' % (s, n)
    return B, theta_opt
