#!/usr/bin/env python2.5
#
# Written (W) 2011-2012 Christian Widmer
# Copyright (C) 2011-2012 Max-Planck-Society

"""
@author: Christian Widmer
@summary: Module that provides a collection of strategies to fit a set
          of ellipses to data.

"""


from collections import defaultdict

import scipy.optimize
import numpy
import loss_functions
import util
from util import Ellipse

import cvxpy
#REGEX: cvxmod --> cvxpy
# % s/cvxpy.optvar(\(.\+\), \(.\))/cvxpy.variable(\2, name=\1)/g
# % s/cvxpy.param(\(.\+\), \(.\))/cvxpy.parameter(\2, name=\1)/g
# % s/cvxpy.param(\(.\+\), \(.\), \(.\))/cvxpy.parameter(\2, \3, name=\1)/g

import sympy


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
    dat = cvxpy.matrix(dat)
    #### parameters

    # data
    X = cvxpy.parameter(N, D, name="X")

    # parameter for eps-insensitive loss
    eps = cvxpy.parameter(1, name="eps")


    #### varibales

    # parameter vector
    theta = cvxpy.variable(D, name="theta")

    # dim = (N x 1)
    s = cvxpy.variable(N, name="s")

    t = cvxpy.variable(N, name="t")

    # simple objective 
    objective = cvxpy.sum(t)
    
    # create problem                                    
    p = cvxpy.program(cvxpy.minimize(objective))
    
    # add constraints 
    # (N x D) * (D X 1) = (N X 1)
    p.constraints.append(X*theta <= s)
    p.constraints.append(-X*theta <= s)
    
    p.constraints.append(s - eps <= t)
    p.constraints.append(0 <= t)
    
    #p.constraints.append(theta[4] == 1)
    # trace constraint
    p.constraints.append(theta[0] + theta[1] == 1)
    
    ###### set values
    X.value = dat
    eps.value = 0.0
    #solver = "mosek" 
    #p.solve(lpsolver=solver)
    p.solve()
    
    cvxpy.printval(theta)

    w = numpy.array(cvxpy.value(theta))
    
    #cvxpy.printval(s)
    #cvxpy.printval(t)

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



def fit_ellipse(x, y):
    """
    fit ellipoid using squared loss and abs loss
    """

    #TODO introduce flag for switching between losses

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
    dat = cvxpy.matrix(dat)
    #### parameters

    # data
    X = cvxpy.parameter(N, D, name="X")


    #### varibales

    # parameter vector
    theta = cvxpy.variable(D, name="theta")

    # simple objective 
    objective = cvxpy.norm1(X*theta)

    # create problem                                    
    p = cvxpy.program(cvxpy.minimize(objective))

    
    p.constraints.append(cvxpy.eq(theta[0,:] + theta[1,:], 1))
   
    ###### set values
    X.value = dat

    p.solve()

    w = numpy.array(theta.value)
    
    #print weights


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




def fit_ellipse_stack(dx, dy, dz, di, norm_type="l2"):
    """
    fit ellipoid using squared loss

    idea to learn all stacks together including smoothness

    """

    #TODO create flag for norm1 vs norm2
    
    assert norm_type in ["l1", "l2", "huber"]

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
    #D = 5
    D = 4

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
        ity = numpy.diag(i) / mean_di

        # dimensionality
        N = len(x)
        d = numpy.zeros((N, D))

        d[:,0] = x*x
        d[:,1] = y*y
        #d[:,2] = x*y
        d[:,2] = x
        d[:,3] = y
        #d[:,4] = numpy.ones(N)

        #d[:,0] = x*x
        #d[:,1] = y*y
        #d[:,2] = x*y
        #d[:,3] = x
        #d[:,4] = y
        #d[:,5] = numpy.ones(N)
    
        # consider intensities
        old_shape = d.shape
        #d = numpy.dot(ity, d)
        assert d.shape == old_shape
    
        print d.shape   
        d = cvxpy.matrix(d)
        #### parameters

        # da
        X = cvxpy.parameter(N, D, name="X" + str(z))
        X.value = d
        X_matrix.append(X)


        #### varibales
    
        # parameter vector
        theta = cvxpy.variable(D, name="theta" + str(z))
        thetas.append(theta)


    # construct obj
    objective = 0

    print "norm type", norm_type 

    for i in xrange(M):


        if norm_type == "l1":
            objective += cvxpy.norm1(X_matrix[i] * thetas[i] + 1.0)
        if norm_type == "l2":
            objective += cvxpy.norm2(X_matrix[i] * thetas[i] + 1.0)

        #TODO these need to be summed
        #objective += cvxpy.huber(X_matrix[i] * thetas[i], 1)
        #objective += cvxpy.deadzone(X_matrix[i] * thetas[i], 1)


    # add smoothness regularization
    reg_const = float(total_N) / float(M-1)

    for i in xrange(M-1):
        objective += reg_const * cvxpy.norm2(thetas[i] - thetas[i+1])


    # create problem                                    
    p = cvxpy.program(cvxpy.minimize(objective))

    prob = p
    import ipdb
    ipdb.set_trace()

    # add constraints
    #for i in xrange(M):
    #    #p.constraints.append(cvxpy.eq(thetas[i][0,:] + thetas[i][1,:], 1))
    #    p.constraints.append(cvxpy.eq(thetas[i][4,:], 1))

    # set solver settings
    p.options['reltol'] = 1e-1
    p.options['abstol'] = 1e-1
    #p.options['feastol'] = 1e-1

    # invoke solver
    p.solve()
    

    # wrap up result
    ellipse_stack = {}

    active_layers = dat.keys()
    assert len(active_layers) == M

    for i in xrange(M):

        w = numpy.array(thetas[i].value)

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
        #bv             = w[2:4]
        bv             = w[2:]
        #c              = w[4]
        c              = 1.0
                
        ## find parameters
        import fit_ellipse
        z, a, b, alpha = fit_ellipse.conic2parametric(A, bv, c)
        print "layer (i,z,a,b,alpha):", i, z, a, b, alpha

        layer = active_layers[i]
        ellipse_stack[layer] = Ellipse(z[0], z[1], layer, a, b, alpha)


    return ellipse_stack




if __name__ == "__main__":

    """
    print "checking gradient of abs loss"
    import fit_circle
    data_x, data_y, data_intensity = fit_circle.load_tif(160, True)
    x = numpy.array(data_x)
    y = numpy.array(data_y)

    print fit_ellipse(x, y)

    print "="*40
    """

    import data_processing
    dx, dy, dz, di, v = data_processing.artificial_data()
    #fit = fit_ellipse_stack_scipy(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack(dx, dy, dz, di)
    fit1 = fit_ellipse_stack(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack_squared(dx, dy, dz, di)


