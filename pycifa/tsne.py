import numpy as np
import matplotlib.pyplot as plt

def tsne(X, labels=None, no_dims=2, initial_dims=None, perplexity=30, verb=False):
    '''
    %      TSNE Performs symmetric t-SNE on dataset X
    %
    % The function performs symmetric t-SNE on the NxD dataset X to reduce its 
    % dimensionality to no_dims dimensions (default = 2). The data is 
    % preprocessed using PCA, reducing the dimensionality to initial_dims 
    % dimensions (default = 30). Alternatively, an initial solution obtained 
    % from an other dimensionality reduction technique may be specified in 
    % initial_solution. The perplexity of the Gaussian kernel that is employed 
    % can be specified through perplexity (default = 30). The labels of the
    % data are not used by t-SNE itself, however, they are used to color
    % intermediate plots. Please provide an empty labels matrix [] if you
    % don't want to plot results during the optimization.
    % The low-dimensional data representation is returned in mappedX.
    %
    % This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7.2b.
    % The toolbox can be obtained from http://homepage.tudelft.nl/19j49
    % You are free to use, change, or redistribute this code in any way you
    % want for non-commercial purposes. However, it is appreciated if you 
    % maintain the name of the original author.
    %
    % (C) Laurens van der Maaten, 2010
    % University California, San Diego / Delft University of Technology
    '''
    if initial_dims is None:
        initial_dims = min(50, X.shape[1])
    # First check whether we already have an initial solution
    if (isinstance(no_dims, np.ndarray)) and (no_dims.size > 1):
        initial_solution = True
        ydata = no_dims.copy()
        no_dims = ydata.shape[1]
        perplexity = initial_dims.copy()
    else:
        initial_solution = False
    # Normalize input data
    X -= np.min(X)
    X /= np.max(X)    
    # Perform preprocessing using PCA
    if not initial_solution:
        if verb:
            print 'Preprocessing data using PCA...'
        if X.shape[1] < X.shape[0]:
            C = np.dot(X.T, X)
        else:
            C = (1. / X.shape[0]) * np.dot(X, X.T)
        M, lambd, _ = np.linalg.svd(C)
        M = M[:, :initial_dims]
        lambd = lambd[ :initial_dims]
        if (X.shape[1] >= X.shape[0]):
            M = np.dot(X.T, M) * (1. / np.sqrt(X.shape[0]*lambd)).T
        X = np.dot(X, M)
        del M, lambd
    # Compute pairwise distance matrix
    sum_X = np.sum(X**2., axis=1, keepdims=True)
    tmp1 = sum_X.T + (-2*np.dot(X, X.T))
    D = sum_X + tmp1
    # Compute joint probabilities
    P, _ = d2p(D, perplexity, 1e-5, verb) # compute affinities using fixed perplexity
    del D
    # Run t-SNE
    if initial_solution:
        ydata = tsne_p(P, labels, ydata)
    else:
        ydata = tsne_p(P, labels, no_dims)
    return ydata
    
def d2p(D, u=15, tol=1e-4, verb=False):
    '''
     D2P Identifies appropriate sigma's to get kk NNs up to some tolerance 
       [P, beta] = d2p(D, kk, tol) 
     Identifies the required precision (= 1 / variance^2) to obtain a Gaussian
     kernel with a certain uncertainty for every datapoint. The desired
     uncertainty can be specified through the perplexity u (default = 15). The
     desired perplexity is obtained up to some tolerance that can be specified
     by tol (default = 1e-4).
     The function returns the final Gaussian kernel in P, as well as the 
     employed precisions per instance in beta.


     (C) Laurens van der Maaten, 2008
     Maastricht University

     This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7.2b.
     The toolbox can be obtained from http://homepage.tudelft.nl/19j49
     You are free to use, change, or redistribute this code in any way you
     want for non-commercial purposes. However, it is appreciated if you 
     maintain the name of the original author.

     (C) Laurens van der Maaten, 2010
     University California, San Diego / Delft University of Technology
    '''    
    # Initialize some variables
    n = D.shape[0] # number of instances
    P = np.zeros([n, n]) # empty probability matrix
    beta = np.ones(n) # empty precision vector
    logU = np.log(u) # log of perplexity (= entropy)
    # Run over all datapoints
    for i in xrange(n):
        if verb:
            if i % 500 == 0:
                print 'Computed P-values %d of %d datapoints...' % (i, n)
        # Set minimum and maximum values for precision
        betamin = -np.inf 
        betamax = np.inf
        # Compute the Gaussian kernel and entropy for the current precision
        tmp = np.hstack([D[i:i+1, :i], D[i:i+1, i+1:]])
        H, thisP = Hbeta(tmp, beta[i])
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while (abs(Hdiff) > tol) and (tries < 50):
            # If not, increase or decrease precision
            if (Hdiff > 0):
                betamin = beta[i]
                if np.isinf(betamax):
                    beta[i] *= 2
                else:
                    beta[i] = (beta[i] + betamax) / 2
            else:
                betamax = beta[i]
                if np.isinf(betamin): 
                    beta[i] /= 2
                else:
                    beta[i] = (beta[i] + betamin) / 2
            # Recompute the values
            tmp = np.hstack([D[i:i+1, :i], D[i:i+1, i+1:]])
            H, thisP = Hbeta(tmp, beta[i])
            Hdiff = H - logU
            tries += 1
        # Set the final row of P
        ind = range(i) + range(i+1, n)
        ind = tuple(ind)
        P[i, ind] = thisP.copy()
    if verb:
        print 'Mean value of sigma: %.5f' % (np.mean(np.sqrt(1. / beta)))
        print 'Minimum value of sigma: %.5f' % (np.min(np.sqrt(1. / beta)))
        print 'Maximum value of sigma: %.5f' % (np.max(np.sqrt(1. / beta)))
    return P, beta
    
def Hbeta(D, beta):
    '''
     Function that computes the Gaussian kernel values given a vector of
     squared Euclidean distances, and the precision of the Gaussian kernel.
     The function also computes the perplexity of the distribution.
    '''
    P = np.exp(-D*beta)
    sumP = np.sum(P)
    H = np.log(sumP) + beta*np.sum(D*P) / sumP
    # why not: H = np.exp(-np.sum(P[P > 1e-5]*np.log(P[P > 1e-5]))) ???
    P /= sumP
    return H, P
    
def tsne_p(P,
    labels=None,
    no_dims=2,
    momentum=0.5,
    final_momentum=0.8,
    mom_switch_iter=250,
    stop_lying_iter=100,
    max_iter=500,
    epsilon=500,
    min_gain=.01
):
    '''
    %TSNE_P Performs symmetric t-SNE on affinity matrix P
    %
    %   mappedX = tsne_p(P, labels, no_dims)
    %
    % The function performs symmetric t-SNE on pairwise similarity matrix P 
    % to create a low-dimensional map of no_dims dimensions (default = 2).
    % The matrix P is assumed to be symmetric, sum up to 1, and have zeros
    % on the diagonal.
    % The labels of the data are not used by t-SNE itself, however, they 
    % are used to color intermediate plots. Please provide an empty labels
    % matrix [] if you don't want to plot results during the optimization.
    % The low-dimensional data representation is returned in mappedX.

    % This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7.2b.
    % The toolbox can be obtained from http://homepage.tudelft.nl/19j49
    % You are free to use, change, or redistribute this code in any way you
    % want for non-commercial purposes. However, it is appreciated if you 
    % maintain the name of the original author.
    %
    % (C) Laurens van der Maaten, 2010
    % University California, San Diego / Delft University of Technology
    
    momentum = 0.5;                                     % initial momentum
    final_momentum = 0.8;                               % value to which momentum is changed
    mom_switch_iter = 250;                              % iteration at which momentum is changed
    stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
    max_iter = 500;                                    % maximum number of iterations
    epsilon = 500;                                      % initial learning rate
    min_gain = .01;                                     % minimum gain for delta-bar-delta
    '''
    realmin = np.finfo(float).tiny
    # First check whether we already have an initial solution
    if (isinstance(no_dims, np.ndarray)) and (no_dims.size > 1):
        initial_solution = True;
        ydata = no_dims.copy()
        no_dims = ydata.shape[1]
    else:
        initial_solution = False
    # Initialize some variables
    n = P.shape[0] # number of instances
    # Make sure P-vals are set properly
    # set diagonal to zero
    np.fill_diagonal(P, 0) 
    # symmetrize P-values
    P = 0.5*(P + P.T)
    # make sure P-values sum to one
    P /= np.sum(P)
    P[P < realmin] = realmin
    # constant in KL divergence
    const = np.sum(P*np.log(P))
    #  lie about the P-vals to find better local minima
    if  not initial_solution:
        P *= 4
    # Initialize the solution
    if not initial_solution:
        ydata = 1e-4*np.random.randn(n, no_dims)
    y_incs  = np.zeros(ydata.shape)
    gains = np.ones(ydata.shape)
    # Run the iterations
    for iter in xrange(max_iter):
        # Compute joint probability that point i and j are neighbors
        sum_ydata = np.sum(ydata**2., axis=1, keepdims=True)
        # Student-t distribution
        num = sum_ydata.T + (-2*np.dot(ydata, ydata.T))
        #bsxfun(@plus, sum_ydata', -2 * (ydata * ydata'))
        num += sum_ydata
        #bsxfun(@plus, sum_ydata, num)
        num = 1./(1 + num) 
        # set diagonal to zero
        np.fill_diagonal(num, 0) 
        # normalize to get probabilities
        Q = np.maximum(num/np.sum(num), realmin)
        # Compute the gradients (faster implementation)
        L = (P - Q)*num
        y_grads = 4. * np.dot(np.diag(np.sum(L, axis=0)) - L, ydata) ## diag#############
        # Update the solution
        #  note that the y_grads are actually -y_grads
        gains = (gains + 0.2)*(np.sign(y_grads) != np.sign(y_incs)) +\
                (gains * 0.8)*(np.sign(y_grads) == np.sign(y_incs))
        gains[gains < min_gain] = min_gain
        y_incs = momentum*y_incs - epsilon*(gains*y_grads)
        ydata += y_incs
        ydata -= np.mean(ydata, axis=0, keepdims=True)
        #ydata = bsxfun(@minus, ydata, mean(ydata, 1));
        # Update the momentum if necessary
        if (iter == mom_switch_iter):
            momentum = final_momentum
        if (iter == stop_lying_iter) and (not initial_solution):
            P /= 4.
        # Print out progress
        if (iter % 10 == 0):
            cost = const - np.sum(P*np.log(Q))
            # disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
        # Display scatter plot (maximally first three dimensions)
        if (iter % 10 == 0) and (labels is not None):
            if no_dims == 1:
                plt.scatter(ydata, ydata, 9) #######, labels, 'filled') ###################3
            elif (no_dims == 2):
                plt.scatter(ydata[:, 0], ydata[:, 1], 9) #####, labels, 'filled') ################
            else:
                fig = plt.figure()
                ax = Axes3D(fig)
                ax.scatter(ydata[:, 0], ydata[:, 1], ydata[:, 2], 40) ######, labels, 'filled') ###################
            #######axis tight
            #######axis off
            plt.show()
    return ydata
