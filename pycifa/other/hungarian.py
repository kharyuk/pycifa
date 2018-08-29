def Hungarian(Perf):
    '''
    % [MATCHING,COST] = Hungarian_New(WEIGHTS)
    %
    % A function for finding a minimum edge weight matching given a MxN Edge
    % weight matrix WEIGHTS using the Hungarian Algorithm.
    %
    % An edge weight of Inf indicates that the pair of vertices given by its
    % position have no adjacent edge.
    %
    % MATCHING return a MxN matrix with ones in the place of the matchings and
    % zeros elsewhere.
    %
    % COST returns the cost of the minimum matching

    % Written by: Alex Melin 30 June 2006
    '''
    # Initialize Variables
    Matching = np.zeros(Perf.shape)
    # Condense the Performance Matrix by removing any unconnected vertices to
    # increase the speed of the algorithm
    # Find the number in each column that are connected
    num_y = np.sum(Perf[abs(Perf) != np.inf], axis=0)
    # Find the number in each row that are connected
    num_x = np.sum(Perf[abs(Perf) != np.inf], axis=1)
    # Find the columns(vertices) and rows(vertices) that are isolated
    x_con = np.where(num_x!=0)[0]
    y_con = np.where(num_y!=0)[0]
    if min(len(x_con), len(y_con)) == 0: #####################
        return None, 0
    # Assemble Condensed Performance Matrix
    P_size = max(len(x_con), len(y_con))
    P_cond = scipy.sparse.coo_matrix([P_size, P_size]) #np.zeros([P_size, P_size])
    P_cond[:len(x_con), :len(y_con)] = Perf[np.ix_(x_con, y_con)]
    # Ensure that a perfect matching exists
    # Calculate a form of the Edge Matrix
    Edge = P_cond.copy()
    Edge[abs(P_cond) != np.inf] = 0
    # Find the deficiency(CNUM) in the Edge Matrix
    cnum = min_line_cover(Edge)
    # Project additional vertices and edges so that a perfect matching exists
    Pmax = np.max(P_cond[abs(P_cond) != np.inf])
    P_size = max(P_cond.shape) + cnum
    P_cond = np.ones([P_size, P_size]) * Pmax
    P_cond[:len(x_con), :len(y_con)] = Perf[np.ix_(x_con,y_con)]
    # MAIN PROGRAM: CONTROLS WHICH STEP IS EXECUTED
    exit_flag = False
    stepnum = 1
    while not exit_flag:
        if stepnum == 1:
            P_cond, stepnum = step1(P_cond)
        elif stepnum == 2:
            r_cov, c_cov, M, stepnum = step2(P_cond)
        elif stepnum == 3:
            c_cov, stepnum = step3(M, P_size)
        elif stepnum == 4:
            M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(P_cond, r_cov, c_cov, M)
        elif stepnum == 5:
            M, r_cov, c_cov, stepnum = step5(M, Z_r, Z_c, r_cov, c_cov)
        elif stepnum == 6:
            P_cond, stepnum = step6(P_cond, r_cov, c_cov)
        elif stepnum == 7:
            exit_flag = True
    # Remove all the virtual satellites and targets and uncondense the
    # Matching to the size of the original performance matrix.
    Matching[np.ix_(x_con, y_con)] = M[:len(x_con), :len(y_con)]
    Cost = np.sum(Perf[Matching==1])
    return Matching, Cost

def step1(P_cond):
    '''
    %   STEP 1: Find the smallest number of zeros in each row
    %           and subtract that minimum from its row
    '''
    rmin = np.min(P_cond, axis=1, keepdims=True)
    P_cond -= rmin
    stepnum = 2
    return P_cond, stepnum

def step2(P_cond):
    '''
    %   STEP 2: Find a zero in P_cond. If there are no starred zeros in its
    %           column or row start the zero. Repeat for each zero
    '''
    # Define variables
    P_size = max(P_cond.shape)
    r_cov = np.zeros([P_size]) # A vector that shows if a row is covered
    c_cov = np.zeros([P_size]) # A vector that shows if a column is covered
    M = np.zeros(P_size) # A mask that shows if a position is starred or primed
    for ii in xrange(P_size):
        for jj in xrange(P_size):
            if (P_cond[ii, jj] == 0) and (r_cov[ii] == 0) and (c_cov[jj] == 0):
                M[ii, jj] = 1
                r_cov[ii] = 1
                c_cov[jj] = 1
    # Re-initialize the cover vectors
    r_cov = np.zeros([P_size])  # A vector that shows if a row is covered
    c_cov = np.zeros([P_size])  # A vector that shows if a column is covered
    stepnum = 3
    return r_cov, c_cov, M, stepnum

def step3(M, P_size):
    '''
    %   STEP 3: Cover each column with a starred zero. If all the columns are
    %           covered then the matching is maximum
    '''
    c_cov = np.sum(M, axis=0)
    if (np.sum(c_cov) == P_size):
        stepnum = 7
    else:
        stepnum = 4
    return c_cov, stepnum
    
def step4(P_cond, r_cov, c_cov, M):
    '''
    %   STEP 4: Find a noncovered zero and prime it.  If there is no starred
    %           zero in the row containing this primed zero, Go to Step 5.
    %           Otherwise, cover this row and uncover the column containing
    %           the starred zero. Continue in this manner until there are no
    %           uncovered zeros left. Save the smallest uncovered value and
    %           Go to Step 6.
    '''
    P_size = max(P_cond.shape)
    zflag = False
    while not zflag:
        # Find the first uncovered zero
        row = 0
        col = 0
        exit_flag = False
        ii = 0
        jj = 0
        while not exit_flag:
            if (P_cond[ii, jj] == 0) and (r_cov[ii] == 0) and (c_cov[jj] == 0):
                row = ii
                col = jj
                exit_flag = True
            jj += 1
            if jj > P_size:
                jj = 0
                ii += 1
            if ii > P_size:
                exit_flag = True
        # If there are no uncovered zeros go to step 6
        if row == 0:
            stepnum = 6
            zflag = True
            Z_r = 0
            Z_c = 0
        else:
            # Prime the uncovered zero
            M[row, col] = 2
            # If there is a starred zero in that row
            # Cover the row and uncover the column containing the zero
            ind = np.where(M[row, :] == 1)[0]
            if (sum(ind) != 0): #######
                r_cov[row] = 1
                zcol = ind.copy()
                c_cov[zcol] = 0
            else:
                stepnum = 5
                zflag = True
                Z_r = row
                Z_c = col
    return M, r_cov, c_cov, Z_r, Z_c, stepnum
    
def step5(M, Z_r, Z_c, r_cov, c_cov):
    '''
    % STEP 5: Construct a series of alternating primed and starred zeros as
    %         follows.  Let Z0 represent the uncovered primed zero found in Step 4.
    %         Let Z1 denote the starred zero in the column of Z0 (if any).
    %         Let Z2 denote the primed zero in the row of Z1 (there will always
    %         be one).  Continue until the series terminates at a primed zero
    %         that has no starred zero in its column.  Unstar each starred
    %         zero of the series, star each primed zero of the series, erase
    %         all primes and uncover every line in the matrix.  Return to Step 3.
    '''

    zflag = False
    ii = 0
    while not zflag:
        # Find the index number of the starred zero in the column
        rindex = np.where(M[:, Z_c[ii]]==1)[0]
        if rindex > 0:
            # Save the starred zero
            ii += 1
            # Save the row of the starred zero
            Z_r[ii] = rindex
            # The column of the starred zero is the same as the column of the
            # primed zero
            Z_c[ii] = Z_c[ii-1]
        else:
            zflag = True
        # Continue if there is a starred zero in the column of the primed zero
        if not zflag:
            # Find the column of the primed zero in the last starred zeros row
            cindex = np.where(M[Z_r[ii], :]==2)[0]
            ii += 1
            Z_r[ii] = Z_r[ii-1]
            Z_c[ii] = cindex
    # UNSTAR all the starred zeros in the path and STAR all primed zeros
    for ii in xrange(len(Z_r)):
        if M[Z_r[ii], Z_c[ii]] == 1:
            M[Z_r[ii], Z_c[ii]] = 0
        else:
            M[Z_r[ii], Z_c[ii]] = 1
    # Clear the covers
    r_cov = r_cov*0
    c_cov = c_cov*0
    # Remove all the primes
    M[M==2] = 0
    stepnum = 3
    return M, r_cov, c_cov, stepnum

def step6(P_cond, r_cov, c_cov):
    '''
    % STEP 6: Add the minimum uncovered value to every element of each covered
    %         row, and subtract it from every element of each uncovered column.
    %         Return to Step 4 without altering any stars, primes, or covered lines.
    '''
    a = np.where(r_cov == 0)[0]
    b = np.where(c_cov == 0)[0]
    minval = np.min(P_cond[np.ix_(a, b)])
    ind1 = np.where(r_cov == 1)[0]
    ind2 = np.where(c_cov == 0)[0]
    P_cond[ind1, :] += minval
    P_cond[:, ind2] -= minval
    stepnum = 4
    return P_cond, stepnum
    
def min_line_cover(Edge):
    # Step 2
    r_cov, c_cov, M, stepnum = step2(Edge)
    # Step 3
    lEd = max(Edge.shape)
    c_cov, stepnum = step3(M, lEd)
    # Step 4
    M, r_cov, c_cov, Z_r, Z_c, stepnum = step4(Edge, r_cov, c_cov, M)
    # Calculate the deficiency
    cnum = lEd - np.sum(r_cov) - np.sum(c_cov)
    return cnum
