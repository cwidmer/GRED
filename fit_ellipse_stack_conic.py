#!/usr/bin/env python2.5
#
# Written (W) 2011-2013 Christian Widmer
# Copyright (C) 2011-2013 Max-Planck-Society

"""
@author: Christian Widmer
@summary: Module that provides a collection of strategies to fit a set
          of ellipses to data.

"""

#solvers.options['feastol'] = 1e-1

from collections import defaultdict
from util import Ellipse

import numpy
import cvxmod


def fit_ellipse_stack(dx, dy, dz, di):
    """
    fit ellipoid beased on data
    """

    # sanity check
    assert len(dx) == len(dy)
    assert len(dx) == len(dz)
    assert len(dx) == len(di)

    # unique zs
    dat = defaultdict(list)

    # resort data
    for idx in range(len(dx)):
        dat[dz[idx]].append( [dx[idx], dy[idx]] )

    # init ret
    ellipse_stack = {}

    for z in dat.keys():
        x_layer = numpy.array(dat[z])[:,0]
        y_layer = numpy.array(dat[z])[:,1]

        # fit ellipse
        try:
            [c, a, b, alpha] = fit_ellipse_squared(x_layer, y_layer)
            #[c, a, b, alpha] = fit_ellipse_linear(x_layer, y_layer)
            #[c, a, b, alpha] = fit_ellipse_eps_insensitive(x_layer, y_layer)
            ellipse_stack[z] = Ellipse(c[0], c[1], z, a, b, alpha)
        except Exception, detail:
            print detail

        #from pprint import pprint
        #pprint( fitellipse(dat_layer, 'linear', constraint = 'bookstein') )
        #pprint( fitellipse(dat_layer, 'linear', constraint = 'trace') )
        #pprint( fitellipse(dat_layer, 'nonlinear') )

        #pprint( fitellipse(dat_layer, 'linear', constraint = 'bookstein') )


    return ellipse_stack



def fitting_obj_stack_gradient(param, dx, dy, dz, di, loss):
    """
    computes objective and gradient for smoothness-regularized least squares ellipse stack
    """


    # centers
    cx = param[0]
    cy = param[1]

    wx = param[2]
    wy = param[3]

    #num_layers = len(set(z))
    #assert len(param) == num_layers+2

    radii = param[4:]

    num_layers = len(radii) / 2

    #print "num_layers", num_layers

    radii_x = radii[0:num_layers]
    radii_y = radii[num_layers:]

    assert len(radii_x) == len(radii_y)

    obj = 0

    gradient_c = [0.0, 0.0]
    gradient_w = [0.0, 0.0]
    gradient_rx = [0.0]*(len(radii_x))
    gradient_ry = [0.0]*(len(radii_y))

    min_z = min(dz)
    mean_di = float(numpy.mean(di))

    for idx in range(len(dx)):

        x = dx[idx]
        y = dy[idx]
        z = dz[idx]

        # determine correct layer
        z_idx = z - min_z

        rx = radii_x[z_idx]
        ry = radii_y[z_idx]

        # normalized contribution
        i = float(di[idx]) / mean_di

        # loss obj and gradient
        obj += i*loss.get_obj(x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_c[0] += i*loss.get_grad("cx", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_c[1] += i*loss.get_grad("cy", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_w[0] += i*loss.get_grad("wx", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_w[1] += i*loss.get_grad("wy", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_rx[z_idx] += i*loss.get_grad("rx", x, y, z, cx, cy, wx, wy, rx, ry)
        gradient_ry[z_idx] += i*loss.get_grad("ry", x, y, z, cx, cy, wx, wy, rx, ry)


    ############################
    # smoothness regularizer using L2-regularization

    reg_smoothness = True

    if reg_smoothness:

        for idx in xrange(num_layers-1):
            
            obj += (radii_x[idx] - radii_x[idx+1])**2
            obj += (radii_y[idx] - radii_y[idx+1])**2

            # compute gradient
            if idx == 0:
                gradient_rx[idx] += 2*radii_x[0] - 2*radii_x[1]
                gradient_ry[idx] += 2*radii_y[0] - 2*radii_y[1]
            else:
                gradient_rx[idx] += 4*radii_x[idx] - 2*radii_x[idx-1] - 2*radii_x[idx+1]
                gradient_ry[idx] += 4*radii_y[idx] - 2*radii_y[idx-1] - 2*radii_y[idx+1]

        # last entry of gradient
        gradient_rx[-1] += 2*radii_x[-1] - 2*radii_x[-2]
        gradient_ry[-1] += 2*radii_y[-1] - 2*radii_y[-2]


    ############################
    # enfore small radii at the ends by means of L1-regularization

    reg_end_param = 10
    num_end_layers = 1

    for idx in xrange(num_end_layers):
        obj += reg_end_param*(radii_x[idx] + radii_x[-idx-1])
        obj += reg_end_param*(radii_y[idx] + radii_y[-idx-1])

        # last entry of gradient
        gradient_rx[idx] += reg_end_param
        gradient_rx[-idx-1] += reg_end_param
        gradient_ry[idx] += reg_end_param
        gradient_ry[-idx-1] += reg_end_param


    # L1-regularize large radii
    #for idx, rx in enumerate(radii_x):
    #    obj += rx
    #    gradient_rx[idx] += 1

    # L1-regularize large radii
    #for idx, ry in enumerate(radii_y):
    #    obj += ry
    #    gradient_ry[idx] += 1

    # L1-regularize large w
    reg_param_w = 5
    obj += wx*reg_param_w
    obj += wy*reg_param_w
    gradient_w[0] += reg_param_w
    gradient_w[1] += reg_param_w


    # build final gradient
    gradient = gradient_c + gradient_w + gradient_rx + gradient_ry

    #return gradient
    return obj, gradient



def fit_ellipse_eps_insensitive(x, y):
    """
    fit ellipoid using epsilon-insensitive loss
    """

    x = numpy.array(x)
    y = numpy.array(y)

    print "shapes", x.shape, y.shape

    assert len(x) == len(y)

    N = len(x)
    D = 5

    dat = numpy.zeros((N, D))
    dat[:,0] = x*x
    dat[:,1] = y*y
    #dat[:,2] = y*x
    dat[:,2] = x
    dat[:,3] = y
    dat[:,4] = numpy.ones(N)


    print dat.shape
    dat = cvxmod.matrix(dat)
    #### parameters

    # data
    X = cvxmod.param("X", N, D)

    # parameter for eps-insensitive loss
    eps = cvxmod.param("eps", 1)


    #### varibales

    # parameter vector
    theta = cvxmod.optvar("theta", D)

    # dim = (N x 1)
    s = cvxmod.optvar("s", N)

    t = cvxmod.optvar("t", N)

    # simple objective 
    objective = cvxmod.sum(t)
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    
    # add constraints 
    # (N x D) * (D X 1) = (N X 1)
    p.constr.append(X*theta <= s)
    p.constr.append(-X*theta <= s)
    
    p.constr.append(s - eps <= t)
    p.constr.append(0 <= t)
    
    #p.constr.append(theta[4] == 1)
    # trace constraint
    p.constr.append(theta[0] + theta[1] == 1)
    
    ###### set values
    X.value = dat
    eps.value = 0.0
    #solver = "mosek" 
    #p.solve(lpsolver=solver)
    p.solve()
    
    cvxmod.printval(theta)

    w = numpy.array(cvxmod.value(theta))
    
    #cvxmod.printval(s)
    #cvxmod.printval(t)

    ## For clarity, fill in the quadratic form variables
    A        = numpy.zeros((2,2))
    A[0,0]   = w[0]
    A.ravel()[1:3] = 0#w[2]
    A[1,1]   = w[1]
    bv       = w[2:4]
    c        = w[4]
    
    ## find parameters
    import fit_ellipse
    z, a, b, alpha = fit_ellipse.conic2parametric(A, bv, c)
    print "XXX", z, a, b, alpha

    return z, a, b, alpha


def fit_ellipse_linear(x, y):
    """
    fit ellipse stack using absolute loss
    """

    x = numpy.array(x)
    y = numpy.array(y)

    print "shapes", x.shape, y.shape

    assert len(x) == len(y)

    N = len(x)
    D = 6

    dat = numpy.zeros((N, D))
    dat[:,0] = x*x
    dat[:,1] = y*y
    dat[:,2] = y*x
    dat[:,3] = x
    dat[:,4] = y
    dat[:,5] = numpy.ones(N)


    print dat.shape
    dat = cvxmod.matrix(dat)


    # norm
    norm = numpy.zeros((N,N))
    for i in range(N):
        norm[i,i] = numpy.sqrt(numpy.dot(dat[i], numpy.transpose(dat[i])))
    norm = cvxmod.matrix(norm)

    #### parameters

    # data
    X = cvxmod.param("X", N, D)
    Q_grad = cvxmod.param("Q_grad", N, N)


    #### varibales
    
    # parameter vector
    theta = cvxmod.optvar("theta", D)
    
    # dim = (N x 1)
    s = cvxmod.optvar("s", N)
    
    # simple objective 
    objective = cvxmod.sum(s)
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    
    # add constraints 
    # (N x D) * (D X 1) = (N x N) * (N X 1)
    p.constr.append(X*theta <= Q_grad*s)
    p.constr.append(-X*theta <= Q_grad*s)
    
    #p.constr.append(theta[4] == 1)
    # trace constraint
    p.constr.append(theta[0] + theta[1] == 1)
    
    ###### set values
    X.value = dat
    Q_grad.value = norm
    #solver = "mosek" 
    #p.solve(lpsolver=solver)
    p.solve()
    
    cvxmod.printval(theta)

    w = numpy.array(cvxmod.value(theta))
    
    #cvxmod.printval(s)
    #cvxmod.printval(t)

    ## For clarity, fill in the quadratic form variables
    A        = numpy.zeros((2,2))
    A[0,0]   = w[0]
    A.ravel()[1:3] = w[2]
    A[1,1]   = w[1]
    bv       = w[3:5]
    c        = w[5]
    
    ## find parameters
    import fit_ellipse
    z, a, b, alpha = fit_ellipse.conic2parametric(A, bv, c)
    print "XXX", z, a, b, alpha

    return z, a, b, alpha



def fit_ellipse_squared(x, y):
    """
    fit ellipoid using squared loss
    """

    assert len(x) == len(y)

    N = len(x)
    D = 5

    dat = numpy.zeros((N, D))
    dat[:,0] = x*x
    dat[:,1] = y*y
    #dat[:,2] = x*y
    dat[:,2] = x
    dat[:,3] = y
    dat[:,4] = numpy.ones(N)


    print dat.shape
    dat = cvxmod.matrix(dat)
    #### parameters

    # data
    X = cvxmod.param("X", N, D)


    #### varibales

    # parameter vector
    theta = cvxmod.optvar("theta", D)

    # simple objective 
    objective = cvxmod.atoms.norm2(X*theta)

    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    p.constr.append(theta[0] + theta[1] == 1)
    
    
    ###### set values
    X.value = dat
    #solver = "mosek" 
    #p.solve(lpsolver=solver)

    pro = p
    #import ipdb
    #ipdb.set_trace()

    p.solve()
    

    w = numpy.array(cvxmod.value(theta))
    
    #print weights
    
    cvxmod.printval(theta)


    ## For clarity, fill in the quadratic form variables
    A              = numpy.zeros((2,2))
    A[0,0]         = w[0]
    A.ravel()[1:3] = 0 #w[2]
    A[1,1]         = w[1]
    bv             = w[2:4]
    c              = w[4]

    ## find parameters
    import fit_ellipse
    z, a, b, alpha = fit_ellipse.conic2parametric(A, bv, c)
    print "XXX", z, a, b, alpha

    return z, a, b, alpha


def fit_ellipse_stack_squared(dx, dy, dz, di):
    """
    fit ellipoid using squared loss

    idea to learn all stacks together including smoothness
    """

    # sanity check
    assert len(dx) == len(dy)
    assert len(dx) == len(dz)
    assert len(dx) == len(di)

    # unique zs
    dat = defaultdict(list)

    # resort data
    for idx in range(len(dx)):
        dat[dz[idx]].append( [dx[idx], dy[idx], di[idx]] )

    # init ret
    ellipse_stack = []
    for idx in range(max(dz)):
        ellipse_stack.append(Ellipse(0, 0, idx, 1, 1, 0))
    

    total_N = len(dx)
    M = len(dat.keys())
    D = 5

    X_matrix = []
    thetas = []


    for z in dat.keys():

        x = numpy.array(dat[z])[:,0]
        y = numpy.array(dat[z])[:,1]

        # intensities
        i = numpy.array(dat[z])[:,2]

        # log intensities
        i = numpy.log(i)

        # create matrix
        ity = numpy.diag(i)

        # dimensionality
        N = len(x)
        d = numpy.zeros((N, D))

        d[:,0] = x*x
        d[:,1] = y*y
        #d[:,2] = x*y
        d[:,2] = x
        d[:,3] = y
        d[:,4] = numpy.ones(N)

        #d[:,0] = x*x
        #d[:,1] = y*y
        #d[:,2] = x*y
        #d[:,3] = x
        #d[:,4] = y
        #d[:,5] = numpy.ones(N)
    
        # consider intensities
        old_shape = d.shape

        d = numpy.dot(ity, d)
        assert d.shape == old_shape
    
        print d.shape   
        d = cvxmod.matrix(d)
        #### parameters

        # da
        X = cvxmod.param("X" + str(z), N, D)
        X.value = d
        X_matrix.append(X)


        #### varibales
    
        # parameter vector
        theta = cvxmod.optvar("theta" + str(z), D)
        thetas.append(theta)




    # contruct objective
    objective = 0
    for (i,X) in enumerate(X_matrix):
        #TODO try abs loss here!
        objective += cvxmod.sum(cvxmod.atoms.square(X*thetas[i]))
        #objective += cvxmod.sum(cvxmod.atoms.abs(X*thetas[i]))

    # add smoothness regularization
    reg_const = float(total_N) / float(M-1)
    for i in xrange(M-1):
        objective += reg_const * cvxmod.sum(cvxmod.atoms.square(thetas[i] - thetas[i+1]))

    print objective

    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))

    # add constraints
    for i in xrange(M):
        p.constr.append(thetas[i][0] + thetas[i][1] == 1)
    
    
    ###### set values
    print "ARRRRRRRRRRRR"
    p.solve()
    

    #print p
    #w = numpy.array(cvxmod.value(thetas))
    #import ipdb
    #ipdb.set_trace()
    #print weights
    #cvxmod.printval(thetas)

    # wrap up result
    ellipse_stack = {}

    active_layers = dat.keys()
    assert len(active_layers) == M


    for i in xrange(M):

        w = numpy.array(cvxmod.value(thetas[i]))

        ## For clarity, fill in the quadratic form variables
        #A        = numpy.zeros((2,2))
        #A[0,0]   = w[0]
        #A.ravel()[1:3] = w[2]
        #A[1,1]   = w[1]
        #bv       = w[3:5]
        #c        = w[5]

        A              = numpy.zeros((2,2))
        A[0,0]         = w[0]
        A.ravel()[1:3] = 0 #w[2]
        A[1,1]         = w[1]
        bv             = w[2:4]
        c              = w[4]
                
        ## find parameters
        import fit_ellipse
        z, a, b, alpha = fit_ellipse.conic2parametric(A, bv, c)
        print "layer (i,z,a,b,alpha):", i, z, a, b, alpha

        layer = active_layers[i]
        ellipse_stack[layer] = Ellipse(z[0], z[1], layer, a, b, alpha)


    return ellipse_stack



def fit_ellipse_stack_abs(dx, dy, dz, di):
    """
    fit ellipoid using squared loss

    idea to learn all stacks together including smoothness
    """

    # sanity check
    assert len(dx) == len(dy)
    assert len(dx) == len(dz)
    assert len(dx) == len(di)

    # unique zs
    dat = defaultdict(list)

    # resort data
    for idx in range(len(dx)):
        dat[dz[idx]].append( [dx[idx], dy[idx], di[idx]] )

    # init ret
    ellipse_stack = []
    for idx in range(max(dz)):
        ellipse_stack.append(Ellipse(0, 0, idx, 1, 1, 0))
    

    total_N = len(dx)
    M = len(dat.keys())
    D = 5

    X_matrix = []
    thetas = []
    slacks = []
    eps_slacks = []

    mean_di = float(numpy.mean(di))

    for z in dat.keys():

        x = numpy.array(dat[z])[:,0]
        y = numpy.array(dat[z])[:,1]

        # intensities
        i = numpy.array(dat[z])[:,2]

        # log intensities
        i = numpy.log(i)

        # create matrix
        ity = numpy.diag(i)# / mean_di

        # dimensionality
        N = len(x)
        d = numpy.zeros((N, D))

        d[:,0] = x*x
        d[:,1] = y*y
        #d[:,2] = x*y
        d[:,2] = x
        d[:,3] = y
        d[:,4] = numpy.ones(N)

        #d[:,0] = x*x
        #d[:,1] = y*y
        #d[:,2] = x*y
        #d[:,3] = x
        #d[:,4] = y
        #d[:,5] = numpy.ones(N)

        print "old", d
        # consider intensities
        old_shape = d.shape
        d = numpy.dot(ity, d)
        print "new", d
        assert d.shape == old_shape
    
        print d.shape   
        d = cvxmod.matrix(d)
        #### parameters

        # da
        X = cvxmod.param("X" + str(z), N, D)
        X.value = d
        X_matrix.append(X)


        #### varibales
    
        # parameter vector
        theta = cvxmod.optvar("theta" + str(z), D)
        thetas.append(theta)


    # construct obj
    objective = 0

    # loss term
    for i in xrange(M):
        objective += cvxmod.atoms.norm1(X_matrix[i] * thetas[i])

    # add smoothness regularization
    reg_const = 5 * float(total_N) / float(M-1)

    for i in xrange(M-1):
        objective += reg_const * cvxmod.norm1(thetas[i] - thetas[i+1])

    #TODO what is this
    #objective += 100*cvxmod.norm1(thetas[0])
    #print objective

    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))

    # add constraints
    """
    for (i,X) in enumerate(X_matrix):
        p.constr.append(X*thetas[i] <= slacks[i])
        p.constr.append(-X*thetas[i] <= slacks[i])

        #eps = 0.5
        #p.constr.append(slacks[i] - eps <= eps_slacks[i])
        #p.constr.append(0 <= eps_slacks[i])
    """

    for i in xrange(1,M):
        p.constr.append(thetas[i][0] + thetas[i][1] == 1.0) # A + C = 1
        #p.constr.append(4 * thetas[i][0] * thetas[i][1] == 1.0) # 4AC - B^2 = 1
        #p.constr.append(thetas[i][4] == 1.0) # F = 1

    # pinch ends
    #TODO not sure how to 
    p.constr.append(thetas[0][0] <= 1.0) # A = 1
    p.constr.append(thetas[0][1] <= 1.0) # C = 1
    p.constr.append(thetas[-1][0] <= 1.0) # A = 1
    p.constr.append(thetas[-1][1] <= 1.0) # C = 1
    
    print p

    ###### set values
    from cvxopt import solvers
    solvers.options['reltol'] = 1e-1
    solvers.options['abstol'] = 1e-1
    print solvers.options

    p.solve()
    
    #print p
    #w = numpy.array(cvxmod.value(thetas))
    #import ipdb
    #ipdb.set_trace()
    #print weights
    #cvxmod.printval(thetas)

    # wrap up result
    ellipse_stack = {}

    active_layers = dat.keys()
    assert len(active_layers) == M


    for i in xrange(M):

        w = numpy.array(cvxmod.value(thetas[i]))

        ## For clarity, fill in the quadratic form variables
        #A        = numpy.zeros((2,2))
        #A[0,0]   = w[0]
        #A.ravel()[1:3] = w[2]
        #A[1,1]   = w[1]
        #bv       = w[3:5]
        #c        = w[5]

        A              = numpy.zeros((2,2))
        A[0,0]         = w[0]
        A.ravel()[1:3] = 0 #w[2]
        A[1,1]         = w[1]
        bv             = w[2:4]
        c              = w[4]
                
        ## find parameters
        import fit_ellipse
        z, a, b, alpha = fit_ellipse.conic2parametric(A, bv, c)
        print "layer (i,z,a,b,alpha):", i, z, a, b, alpha

        layer = active_layers[i]
        ellipse_stack[layer] = Ellipse(z[0], z[1], layer, a, b, alpha)


    return ellipse_stack




if __name__ == "__main__":

    """
    import fit_circle
    data_x, data_y, data_intensity = fit_circle.load_tif(160, True)
    x = numpy.array(data_x)
    y = numpy.array(data_y)

    print fit_ellipse_squared(x, y)
    """

    print "="*40

    import data_processing
    dx, dy, dz, di, v = data_processing.artificial_data()
    #fit = fit_ellipse_stack_scipy(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack_abs(dx, dy, dz, di)
    fit1 = fit_ellipse_stack_squared(dx, dy, dz, di)

    

