import numpy as np
import scipy.sparse
import scipy.optimize

def accuracy(true_labels, cluster_labels):
    # ACCURACY Compute clustering accuracy using the true and cluster labels and
    #   return the value in 'score'.
    #
    #   Input  : true_labels    : N-by-1 vector containing true labels
    #            cluster_labels : N-by-1 vector containing cluster labels
    #
    #   Output : score          : clustering accuracy

    # Compute the confusion matrix 'cmat', where
    #   col index is for true label (CAT),
    #   row index is for cluster label (CLS).
    n = len(true_labels)
    #cat = spconvert([(1:n)' true_labels ones(n,1)]);
    cat = scipy.sparse.coo_matrix((np.ones(n), (np.arange(n), true_labels)))
    cls = scipy.sparse.coo_matrix((np.ones(n), (np.arange(n), cluster_labels)))
    cmat = cls.T
    cmat = cmat.dot(cat)
    #cmat = np.dot(cls.T, cat)
    cmat = -cmat.todense()
    # Calculate accuracy
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cmat) # hungarian
    cost = cmat[row_ind, col_ind].sum()
    score = 100.*(-cost/n)
    return score
    
def CalcSIR(A, Aest, returnMaps=False):
    '''
    Original matlab implementation is written by Anh-Huy Phan:
    http://www.bsp.brain.riken.jp/~phan/index.html
    '''
    eps = np.spacing(1)
    a = A.copy()#real
    b = Aest.copy()#real
    # demean
    a -= np.mean(a, axis=0, keepdims=True)
    b -= np.mean(b, axis=0, keepdims=True)
    # normalize
    a /= (np.linalg.norm(a, axis=0, keepdims=True) + eps)
    b /= (np.linalg.norm(b, axis=0, keepdims=True) + eps)
    
    col1 = a.shape[1]
    col2 = b.shape[1]
    flag = np.zeros(col1)
    maps = np.zeros(col2)
    MSE = np.inf * np.ones(col2)
    for i in xrange(col2):
        #tmp1 = bsxfun( @minus, Aest( :, i ), A )
        tmp1 = b[:, i:i+1] - a # [.., col1]
        #tmp2 = bsxfun( @plus, Aest( :, i ), A )
        tmp2 = b[:, i:i+1] + a # [.., col1]
        temp = np.minimum(np.sum(tmp1**2, axis=0), np.sum(tmp2**2, axis=0)) # [col1]
        temp = np.maximum(temp, flag) # -> flag has _col1_ length
        maps[i] = np.argmin(temp) # i -> maps has col2 shape
        MSE[i] = np.min(temp) # i -> MSE has col2 shape
        flag[int(maps[i])] = np.inf # -> col1 length
    SIR = -10.*np.log10(MSE)
    if returnMaps:
        return MSE, maps
    return SIR
    
def CalcSIR_custom(a_in, b_in):
    '''
    Unfortunately, it is valid only for two signals.
    If a_in or b_in contain more than 1 signal, behaviour of function is invalid
    '''

    # demean
    a = a_in - np.mean(a_in, axis=0, keepdims=True)
    b = b_in - np.mean(b_in, axis=0, keepdims=True)
    # normalize
    a /= np.linalg.norm(a_in, axis=0, keepdims=True)
    b /= np.linalg.norm(b_in, axis=0, keepdims=True)
    # whiten
    #a = whiten(a)
    #b = whiten(b)
    #a /= a.var(axis=0, keepdims=True, ddof=1)**0.5
    #b /= b.var(axis=0, keepdims=True, ddof=1)**0.5
    #print a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)
    # Signal-to-interface ratio (SIR)
    sir = (a**2.).sum(axis=0) / ((a-b)**2.).sum(axis=0)###################
    # dB scale
    sir = 10.*np.log10(sir)
    return sir
    
def MutualInfo(L1_in, L2_in):
    '''
    %   mutual information
    % http://www.zjucadcg.cn/dengcai/Data/data.html
    %  @ARTICLE{CHH05,
    %         AUTHOR = {Deng Cai and Xiaofei He and Jiawei Han},
    %         TITLE = {Document Clustering Using Locality Preserving Indexing},
    %         JOURNAL = {IEEE Transactions on Knowledge and Data Engineering},
    %         YEAR = {2005},
    %         volume = {17},
    %         number = {12},
    %         pages = {1624-1637},
    %         month = {December},}

    %===========    
    '''
    L1 = L1_in.copy()
    L2 = L2_in.copy()
    assert L1.shape == L2.shape, 'shape of L1 must be equal to shape of L2'
    Label = set(L1)
    Label = list(Label)
    nClass = len(Label)
    Label2 = set(L2)
    Label2 = list(Label2)
    nClass2 = len(Label2)
    if nClass2 < nClass:
         # smooth
         tmp = np.zeros(L1.size + len(Label))
         tmp[:L1.size] += L1
         tmp[L1.size:] += np.array(Label)
         L1 = tmp.copy()
         tmp = np.zeros(L2.size + len(Label))
         tmp[:L2.size] += L2
         tmp[L2.size:] += np.array(Label)
         L2 = tmp.copy()
    elif nClass2 > nClass:
         # smooth
         tmp = np.zeros(L1.size + len(Label2))
         tmp[:L1.size] += L1
         tmp[L1.size:] += np.array(Label2)
         L1 = tmp.copy()
         tmp = np.zeros(L2.size + len(Label2))
         tmp[:L2.size] += L2
         tmp[L2.size:] += np.array(Label2)
         L2 = tmp.copy()
    G = np.zeros([nClass, nClass])
    for i in xrange(nClass):
        for j in xrange(nClass):
            G[i, j] = np.sum((L1 == Label[i]) * (L2 == Label[j]))
    sumG = np.sum(G)
    P1 = np.sum(G, axis=1, keepdims=True) / sumG
    P2 = np.sum(G, axis=0, keepdims=True) / sumG
    if (np.sum(P1==0) > 0) or (np.sum(P2==0) > 0):
        # smooth
        raise ValueError('Smooth fail!') ####################
    else:
        H1 = np.sum(-P1*np.log2(P1))
        H2 = np.sum(-P2*np.log2(P2))
        P12 = G/sumG
        PPP = P12 / np.tile(P2, (nClass,1))
        PPP /= np.tile(P1, (1, nClass))
        PPP[abs(PPP) < 1e-12] = 1
        MI = np.sum(P12 * np.log2(PPP))
        MIhat = MI / max(H1, H2)
        ##########   why complex ?    ###########
        MIhat = MIhat.real
    return MIhat
