import numpy as np
import scipy.sparse

from utils import EuDist2
from utils import sub2ind
from utils import vec
from utils import NormalizeFea
from utils import reshape


def constructW(
        fea,
        bNormalized=False,
        NeighborMode='KNN',
        kKNN=5,
        bLDA=False,
        bSelfConnected=False,
        gnd=None,
        WeightMode='HeatKernel',
        t=1,
        bTrueKNN=False,
        SameCategoryWeight=1,
        bSemiSupervised=False,
        semiSplit=None
):
    '''
    %	Usage:
    %	W = constructW(fea,options)
    %
    %	fea: Rows of vectors of data points. Each row is x_i
    %   options: Struct value in Matlab. The fields in options that can be set:
    %                  
    %           NeighborMode -  Indicates how to construct the graph. Choices
    %                           are: [Default 'KNN']
    %                'KNN'            -  k = 0
    %                                       Complete graph
    %                                    k > 0
    %                                      Put an edge between two nodes if and
    %                                      only if they are among the k nearst
    %                                      neighbors of each other. You are
    %                                      required to provide the parameter k in
    %                                      the options. Default k=5.
    %               'Supervised'      -  k = 0
    %                                       Put an edge between two nodes if and
    %                                       only if they belong to same class. 
    %                                    k > 0
    %                                       Put an edge between two nodes if
    %                                       they belong to same class and they
    %                                       are among the k nearst neighbors of
    %                                       each other. 
    %                                    Default: k=0
    %                                   You are required to provide the label
    %                                   information gnd in the options.
    %                                              
    %           WeightMode   -  Indicates how to assign weights for each edge
    %                           in the graph. Choices are:
    %               'Binary'       - 0-1 weighting. Every edge receiveds weight
    %                                of 1. 
    %               'HeatKernel'   - If nodes i and j are connected, put weight
    %                                W_ij = exp(-norm(x_i - x_j)/2t^2). You are 
    %                                required to provide the parameter t. [Default One]
    %               'Cosine'       - If nodes i and j are connected, put weight
    %                                cosine(x_i,x_j). 
    %               
    %            k         -   The parameter needed under 'KNN' NeighborMode.
    %                          Default will be 5.
    %            gnd       -   The parameter needed under 'Supervised'
    %                          NeighborMode.  Colunm vector of the label
    %                          information for each data point.
    %            bLDA      -   0 or 1. Only effective under 'Supervised'
    %                          NeighborMode. If 1, the graph will be constructed
    %                          to make LPP exactly same as LDA. Default will be
    %                          0. 
    %            t         -   The parameter needed under 'HeatKernel'
    %                          WeightMode. Default will be 1
    %         bNormalized  -   0 or 1. Only effective under 'Cosine' WeightMode.
    %                          Indicates whether the fea are already be
    %                          normalized to 1. Default will be 0
    %      bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 0
    %                          if 'Supervised' NeighborMode & bLDA == 1,
    %                          bSelfConnected will always be 1. Default 0.
    %            bTrueKNN  -   0 or 1. If 1, will construct a truly kNN graph
    %                          (Not symmetric!). Default will be 0. Only valid
    %                          for 'KNN' NeighborMode
    %
    %
    %    Examples:
    %
    %       fea = rand(50,15);
    %       options = [];
    %       options.NeighborMode = 'KNN';
    %       options.k = 5;
    %       options.WeightMode = 'HeatKernel';
    %       options.t = 1;
    %       W = constructW(fea,options);
    %       
    %       
    %       fea = rand(50,15);
    %       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
    %       options = [];
    %       options.NeighborMode = 'Supervised';
    %       options.gnd = gnd;
    %       options.WeightMode = 'HeatKernel';
    %       options.t = 1;
    %       W = constructW(fea,options);
    %       
    %       
    %       fea = rand(50,15);
    %       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
    %       options = [];
    %       options.NeighborMode = 'Supervised';
    %       options.gnd = gnd;
    %       options.bLDA = 1;
    %       W = constructW(fea,options);      
    %       
    %
    %    For more details about the different ways to construct the W, please
    %    refer:
    %       Deng Cai, Xiaofei He and Jiawei Han, "Document Clustering Using
    %       Locality Preserving Indexing" IEEE TKDE, Dec. 2005.
    %    
    %
    %    Written by Deng Cai (dengcai2 AT cs.uiuc.edu), April/2004, Feb/2006,
    %                                             May/2007
    '''
    bSpeed  = 1
    NeighborMode_low = NeighborMode.lower()
    if NeighborMode_low == 'KNN'.lower():
        # For simplicity, we include the data point itself in the kNN
        k = kKNN
    elif NeighborMode_low == 'Supervised'.lower():
        k = kLDA # 0
        assert gnd is not None, 'Label(gnd) should be provided under ''Supervised'' NeighborMode!'
        if (fea is not None) and (len(gnd) != fea.shape[0]):
            raise ValueError('gnd doesn''t match with fea!')
    else:
        raise NotImplementedError('NeighborMode does not exist!')
    bBinary = False
    bCosine = False
    WeightMode_low = WeightMode.lower()
    if WeightMode_low == 'Binary'.lower():
        bBinary = True
    elif WeightMode_low == 'HeatKernel'.lower():
        if t is None:
            nSmp = fea.shape[0]
            if nSmp > 3000:
                tmpInd = np.random.randint(0, nSmp, size=[3000])
                D = EuDist2(fea[tuple(tmpInd), :])
            else:
                D = EuDist2(fea)
            t = np.mean(D)
    elif WeightMode_low == 'Cosine'.lower():
        bCosine = True
    else:
        raise ValueError('WeightMode does not exist!')
    if gnd is not None:
        nSmp = len(gnd)
    else:
        nSmp = fea.shape[0]
    maxM = 62500000 # 500M
    BlockSize = int(np.floor(float(maxM) / (nSmp*3)))
    if NeighborMode_low == 'Supervised'.lower():
        Label = set(gnd)
        nLabel = len(Label)
        if bLDA:
            G = np.zeros([nSmp, nSmp])
            for idx in xrange(nLabel):
                classIdx = gnd == Label[idx]
                G[np.ix_(classIdx, classIdx)] = 1./np.sum(classIdx)
            W = scipy.sparse.csr_matrix(G) #########################
            return G
        if WeightMode_low == 'Binary'.lower():
            if k > 0:
                G = np.zeros([nSmp*(k+1), 3])
                idNow = 0
                for i in xrange(nLabel):
                    classIdx = np.where(gnd == Label[i])
                    D = EuDist2(fea[classIdx, :], None, False)
                    idx = np.argsort(D, axis=1)
                    dump = np.sort(D, axis=1) # sort each row
                    del D, dump
                    idx = idx[:, :k+1]
                    nSmpClass = len(classIdx)*(k+1)
                    G[idNow : nSmpClass+idNow, 0] = np.tile(classIdx, [k+1, 1])
                    G[idNow : nSmpClass+idNow, 1] = classIdx[idx]
                    G[idNow : nSmpClass+idNow, 2] = 1
                    idNow = idNow + nSmpClass
                    del idx
                G = scipy.sparse.coo_matrix((G[:, 0], [G[:, 1], G[:, 2]]), (nSmp, nSmp)) ######
                G = G.tolil()
                #G = G.todense()
                G = np.maximum(G, G.T)
            else:
                G = np.zeros([nSmp, nSmp])
                for i in xrange(nLabel):
                    classIdx = np.where(gnd == Label[i])
                    G[np.ix_(classIdx, classIdx)] = 1.
            if not bSelfConnected:
                if isinstance(G, np.ndarray):
                    np.fill_diagonal(G, 0)
                else:
                    G.setdiag(0)
            W = scipy.sparse.coo_matrix(G) ####################################
            W = W.tolil()
        elif WeightMode_low == 'HeatKernel'.lower():
            if k > 0:
                G = np.zeros([nSmp*(k+1), 3])
                idNow = 0
                for i in xrange(nLabel):
                    classIdx = np.where(gnd == Label[i])
                    D = EuDist2(fea[classIdx, :], None, False)
                    idx = np.argsort(D, axis=1)
                    dump = np.sort(D, axis=1) # sort each row
                    del D
                    idx = idx[:, :k+1]
                    dump = dump[:, :k+1]
                    dump = np.exp(-dump/(2.*(t**2.)))
                    nSmpClass = len(classIdx)*(k+1)
                    G[idNow : nSmpClass + idNow, 0] = np.tile(classIdx, [k+1, 1])
                    G[idNow : nSmpClass + idNow, 1] = classIdx(vec(idx))
                    G[idNow : nSmpClass + idNow, 2] = vec(dump)
                    idNow += nSmpClass
                    del dump, idx
                G = scipy.sparse.coo_matrix((G[:, 0], [G[:, 1], G[:, 2]]), (nSmp, nSmp))
                G = G.tolil()
            else:
                G = np.zeros([nSmp, nSmp])
                for i in xrange(nLabel):
                    classIdx = np.where(gnd == Label[i])
                    D = EuDist2(fea[classIdx, :], None, False) ########################################################
                    D = exp(-D / (2*(t**2)))
                    G[np.ix_(classIdx, classIdx)] = D
            if not bSelfConnected:
                if isinstance(G, np.ndarray):
                    np.fill_diagonal(G, 0)
                else:
                    G.setdiag(0)
            W = scipy.sparse.coo_matrix(np.maximum(G, G.T)) ######################
            W = W.tolil()
        elif WeightMode_low == 'Cosine'.lower():
            if not bNormalized:
                fea = NormalizeFea(fea)
            if k > 0:
                G = np.zeros([nSmp*(k+1), 3])
                idNow = 0
                for i in xrange(nLabel):
                    classIdx = np.where(gnd == Label[i])
                    D = np.dot(fea[classIdx, :], fea[classIdx, :].T)
                    idx = np.argsort(-D, axis=1)
                    dump = np.sort(-D, axis=1) # sort each row
                    del D
                    idx = idx[:, :k+1]
                    dump = -dump[:, :k+1]
                    nSmpClass = len(classIdx)*(k+1)
                    G[idNow : nSmpClass + idNow, 0] = np.tile(classIdx, [k+1, 1])
                    G[idNow : nSmpClass + idNow, 1] = classIdx(vec(idx))
                    G[idNow : nSmpClass + idNow, 2] = vec(dump)
                    idNow += nSmpClass
                    del dump, idx
                G = scipy.sparse.coo_matrix((G[:, 0], [G[:, 1], G[:, 2]]), (nSmp, nSmp))
                G = G.tolil()
            else:
                G = np.zeros([nSmp, nSmp])
                for i in xrange(nLabel):
                    classIdx = np.where(gnd == Label[i])
                    G[np.ix_(classIdx, classIdx)] = np.dot(fea[classIdx, :],
                                                           fea[classIdx, :].T)
            if not bSelfConnected:
                if isinstance(G, np.ndarray):
                    np.fill_diagonal(G, 0)
                else:
                    G.setdiag(0)
            if isinstance(G, np.ndarray):
                W = np.maximum(G, G.T)
                W = scipy.sparse.csr_matrix(W)
            else:
                W = G.maximum(G.T)
                W = W.tolil()
        else:
            raise ValueError('WeightMode does not exist!')
        return W

    if (bCosine) and (not bNormalized):
        Normfea = NormalizeFea(fea)
    if (NeighborMode.lower() == 'KNN'.lower()) and (k > 0):
        if not (bCosine and bNormalized):
            G = np.zeros([nSmp*(k+1), 3])
            for i in xrange(int(np.ceil(float(nSmp) / BlockSize))):
                if (i == int(np.ceil(float(nSmp) / BlockSize))-1):
                    smpIdx = range(i*BlockSize, nSmp)
                    dist = EuDist2(fea[smpIdx, :], fea, False)
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros([nSmpNow, k+1])
                        idx = dump.copy()
                        for j in xrange(k+1):
                            dump[:, j] = np.min(dist, axis=1)
                            idx[:, j] = np.argmin(dist, axis=1)
                            temp = idx[:, j]*nSmpNow + np.arange(nSmpNow)
                            temp = temp.astype(np.int)
                            shapeDist = dist.shape
                            dist = dist.flatten(order='F')
                            dist[temp] = 1e100 ##############################
                            dist = reshape(dist, shapeDist)
                    else:
                        idx = np.argsort(dist, axis=1)
                        dump = np.sort(dist, axis=1) # sort each row
                        idx = idx[:, :k+1]
                        dump = dump[:, :k+1]
                    if not bBinary:
                        if bCosine:
                            dist = np.dot(Normfea[smpIdx, :], Normfea.T)
                            dist = dist.todense() #########################################
                            linidx = range(idx.shape[0])
                            ##dump = dist(sub2ind(size(dist), linidx(:,ones(1,size(idx,2))), idx));
                            dumpInd = sub2ind(dist.shape, [linidx[:, tuple([1]*idx.shape[1]), idx]])
                            dump = dist[dumpInd]
                        else:
                            dump = np.exp(-dump/(2*(t**2)))
                    G[i*BlockSize*(k+1) : nSmp*(k+1), 0] = np.array(smpIdx*(k+1))
                    G[i*BlockSize*(k+1) : nSmp*(k+1), 1] = vec(idx)
                    if not bBinary:
                        G[i*BlockSize*(k+1) : nSmp*(k+1), 2] = vec(dump)
                    else:
                        G[i*BlockSize*(k+1) : nSmp*(k+1), 2] = 1
                else:
                    smpIdx = range(i*BlockSize, (i+1)*BlockSize)
                    dist = EuDist2(fea[smpIdx, :], fea, False)
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros([nSmpNow, k+1])
                        idx = dump.copy()
                        for j in xrange(k+1):
                            idx[:, j] = np.argmin(dist, axis=1)
                            dump[:, j] = np.min(dist, axis=1)
                            temp = idx[:, j]*nSmpNow + np.arange(nSmpNow)
                            dist[tuple(temp)] = 1e100
                    else:
                        idx = np.argsort(dist, axis=1)
                        dump = np.sort(dist, axis=1) # sort each row
                        idx = idx[:, :k+1]
                        dump = dump[:, :k+1]
                    if  not bBinary:
                        if bCosine:
                            dist = np.dot(Normfea[smpIdx, :], Normfea.T)
                            dist = dist.todense() ################################################33
                            linidx = range(idx.shape[0])
                            dumpInd = sub2ind(dist.shape, [np.tile(linidix, [1, idx.shape[1]]), idx])
                            dump = dist[dumpInd]
                        else:
                            dump = np.exp(-dump / (2*(t**2)))
                    G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 0] = np.tile(smpIdx, [k+1,1]) ##################
                    G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 1] = vec(idx)
                    if not bBinary:
                        G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 2] = vec(dump)
                    else:
                        G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 2] = 1
            W = scipy.sparse.coo_matrix((G[:, 0], [G[:, 1], G[:, 2]]), (nSmp, nSmp))
            W = W.tolil()
        else:
            G = np.zeros([nSmp*(k+1), 3])
            for i in xrange(int(np.ceil(float(nSmp) / BlockSize))):
                if (i == int(np.ceil(float(nSmp) / BlockSize))-1):
                    smpIdx = range(i*BlockSize, nSmp)
                    dist = np.dot(fea[tuple(smpIdx), : ], fea.T)
                    dist = dist.todense() ########################################################################3
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros([nSmpNow, k+1])
                        idx = dump.copy()
                        for j in xrange(k+1):
                            idx[:, j] = np.argmax(dist, axis=1)
                            dump[:, j] = np.max(dist, axis=1)
                            temp = idx[:, j]*nSmpNow + np.arange(nSmpNow)
                            dist[tuple(temp)] = 0
                    else:
                        idx = np.argsort(-dist, axis=1)
                        dump = np.sort(-dist, axis=1) # sort each row
                        idx = idx[:, :k+1]
                        dump = -dump[:, :k+1]
                    G[i*BlockSize*(k+1) : nSmp*(k+1), 0] = np.tile(smpIdx, [1, k+1]) ##################################
                    G[i*BlockSize*(k+1) : nSmp*(k+1), 1] = vec(idx)
                    G[i*BlockSize*(k+1) : nSmp*(k+1), 2] = vec(dump)
                else:
                    smpIdx = range(i*BlockSize, (i+1)*BlockSize)
                    dist = np.dot(fea[tuple(smpIdx), :], fea.T)
                    dist = dist.todense() ######
                    if bSpeed:
                        nSmpNow = len(smpIdx)
                        dump = np.zeros([nSmpNow, k+1])
                        idx = dump.copy()
                        for j in xrange(k+1):
                            idx[:, j] = np.argmax(dist, axis=1)
                            dump[:, j] = np.max(dist, axis=1)
                            temp = idx[:, j]*nSmpNow + np.arange(nSmpNow)
                            dist[tuple(temp)] = 0
                    else:
                        idx = np.argsort(-dist, axis=1)
                        dump = np.sort(-dist, axis=1) # sort each row
                        idx = idx[:, :k+1]
                        dump = -dump[:, :k+1]
                    G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 0] = np.tile(smpIdx, [1, k+1])
                    G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 1] = vec(idx)
                    G[i*BlockSize*(k+1) : (i+1)*BlockSize*(k+1), 2] = vec(dump)
            W = scipy.sparse.coo_matrix((G[:, 0], [G[:, 1], G[:, 2]]), (nSmp, nSmp))
            W = W.tolil()
        if bBinary:
            W[W != 0] = 1
        
        if bSemiSupervised:
            tmpgnd = gnd(semiSplit)
            Label = set(tmpgnd)
            nLabel = len(Label)
            G = np.zeros([np.sum(semiSplit), np.sum(semiSplit)])
            for idx in xrange(nLabel):
                classIdx = tmpgnd == Label[idx]
                G[np.ix_(classIdx, classIdx)] = 1
            Wsup = scipy.sparse.coo_matrix(G) ############################################################
            Wsup = Wsup.tolil()
            W[np.ix_(semiSplit, semiSplit)] = (Wsup > 0)*SameCategoryWeight ##############################
        if not bSelfConnected:
            if isinstance(W, np.ndarray):
                np.fill_diagonal(W, 0)
            else:
                W.setdiag(0)
        if not bTrueKNN:
            if isinstance(W, np.ndarray):
                W = np.maximum(W, W.T)
                W = scipy.sparse.csr_matrix(W)
            else:
                W = W.maximum(W.T)
                W = W.tolil()
        return W
    # strcmpi(options.NeighborMode,'KNN') & (options.k == 0)
    # Complete Graph
    if WeightMode_low == 'Binary'.lower():
        raise ValueError('Binary weight can not be used for complete graph!')
    elif WeightMode_low == 'HeatKernel'.lower():
        W = EuDist2(fea, bSqrt=False)
        W = np.exp(-W / (2*(t**2)))
    elif WeightMode_low == 'Cosine'.lower():
        W = np.dot(Normfea, Normfea.T)
    else:
        raise ValueError('WeightMode does not exist!')
    if not bSelfConnected:
        if isinstance(W, np.ndarray):
            np.fill_diagonal(W, 0)
        else:
            W.setdiag(0)
    if isinstance(W, np.ndarray):
        W = np.maximum(W, W.T)
        W = scipy.sparse.csr_matrix(W)
    else:
        W = W.maximum(W.T)
        W = W.tolil()
    return W

