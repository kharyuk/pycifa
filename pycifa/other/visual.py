

def visual(W_in, mag, cols, ysize=None, inverse=False):
    '''
    % visual - display a basis for image patches
    %
    % W        the basis, with patches as column vectors
    % mag      magnification factor
    % cols     number of columns (x-dimension of map)
    % ysize    [optional] height of each subimage
    %
    ''' 
    W = W_in.copy()
    # Is the basis non-negative    
    if (np.min(W) >= 0):
        # Make zeros white and positive values darker, as in earlier NMF papers
        W = -W
        maxi = 0
        mini = np.min(W)
        bgval = mini/2
    else:
        # Make zero gray, positive values white, and negative values black
        maxi = np.max(abs(W))
        mini = -maxi
        bgval = maxi
    # Get maximum absolute value (it represents white or black; zero is gray)
    # This is the side of the window
    if ysize is None:
        ysize = np.sqrt(W.shape[0])
    xsize = W.shape[0] / ysize
    # Helpful quantities
    xsizem = xsize-1
    xsizep = xsize+1
    ysizem = ysize-1
    ysizep = ysize+1
    rows = [None]*(W.shape[1] / cols)
    # Initialization of the image
    I = bgval*np.ones([2 + ysize*rows+rows - 1, 2 + xsize*cols+cols - 1])
    for i in xrange(rows):
        for j in xrange(cols):
            if (i*cols + j + 1) > W.shape[1]:
                # This leaves it at background color
                pass  
            else:
                # This sets the patch
                I[i*ysizep + 2 : i*ysizep + ysize + 1,
	              j*xsizep + 2 : j*xsizep + xsize + 1] = reshape(W[:,i*cols+j+1],
	                                                             [ysize, xsize])
    # Make a black border
    I[0, :] = mini
    I[:, 0] = mini
    I[-1, :] = mini
    I[:, -1] = mini
    I = scipy.ndimage.interpolation.zoom(I, mag)

    #colormap(gray(256));
    #iptsetpref('ImshowBorder','tight'); 
    #subplot('position',[0,0,1,1]);
    if inverse:
        plt.imshow(imcomplement(I), extent=[0, 1, 0, 1], cmap=plt.cm.gray)
    else:
        plt.imshow(I, vmin=mini, vmax=maxi, cmap=plt.cm.gray)
    #truesize;  
    plt.show()
