#!/usr/bin/env python2.5
#
# Written (W) 2011-2013 Christian Widmer
# Copyright (C) 2011-2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@summary: Procedures for fitting stacks of ellipses using 
          conic parameterization

"""

#solvers.options['feastol'] = 1e-1

from collections import defaultdict

import numpy
import cvxmod

from util import Ellipse, conic_to_ellipse


def fit_ellipse_stack_decoupled(dx, dy, dz, di):
    """
    fit ellipse stack by independently fitting
    each layer
    """

    # sanity check
    assert len(dx) == len(dy) == len(dz) == len(di)

    # unique zs
    dat = defaultdict(list)

    # resort data by layer
    for idx in range(len(dx)):
        dat[dz[idx]].append( [dx[idx], dy[idx]] )

    # init ret
    ellipse_stack = {}

    # iterate over layers
    for z in dat.keys():
        x_layer = numpy.array(dat[z])[:, 0]
        y_layer = numpy.array(dat[z])[:, 1]

        # fit ellipse
        try:
            [c, a, b, alpha] = fit_ellipse_squared(x_layer, y_layer)
            #[c, a, b, alpha] = fit_ellipse_linear(x_layer, y_layer)
            #[c, a, b, alpha] = fit_ellipse_eps_insensitive(x_layer, y_layer)

            # reconstruct ellipse stack
            ellipse_stack[z] = Ellipse(c[0], c[1], z, a, b, alpha)

        except Exception, detail:
            print detail

    return ellipse_stack



def fit_ellipse_eps_insensitive(x, y):
    """
    fit ellipse using epsilon-insensitive loss
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
    theta_ = numpy.array(cvxmod.value(theta))
    ellipse = conic_to_ellipse(theta_)

    return ellipse


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
    theta_ = numpy.array(cvxmod.value(theta))
    ellipse = conic_to_ellipse(theta_)

    return ellipse


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

    p.solve()
    
    cvxmod.printval(theta)

    theta_ = numpy.array(cvxmod.value(theta))
    ellipse = conic_to_ellipse(theta_)

    return ellipse



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
    p.solve()
    

    # wrap up result
    ellipse_stack = {}

    active_layers = dat.keys()
    assert len(active_layers) == M

    for i in xrange(M):

        theta_ = numpy.array(cvxmod.value(thetas[i]))
        z_layer = active_layers[i]
        ellipse_stack[z_layer] = conic_to_ellipse(theta_)
        ellipse_stack[z_layer].cz = z_layer

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


    # create problem                                    
    prob = cvxmod.problem(cvxmod.minimize(objective))

    # add constraints
    """
    for (i,X) in enumerate(X_matrix):
        p.constr.append(X*thetas[i] <= slacks[i])
        p.constr.append(-X*thetas[i] <= slacks[i])

        #eps = 0.5
        #p.constr.append(slacks[i] - eps <= eps_slacks[i])
        #p.constr.append(0 <= eps_slacks[i])
    """

    # add non-degeneracy constraints
    for i in xrange(1, M-1):
        prob.constr.append(thetas[i][0] + thetas[i][1] == 1.0) # A + C = 1

    # pinch ends
    prob.constr.append(cvxmod.sum(thetas[0]) >= -0.01)
    prob.constr.append(cvxmod.sum(thetas[-1]) >= -0.01)

    print prob

    ###### set values
    from cvxopt import solvers
    solvers.options['reltol'] = 1e-1
    solvers.options['abstol'] = 1e-1
    print solvers.options

    prob.solve()
    

    # wrap up result
    ellipse_stack = {}

    active_layers = dat.keys()
    assert len(active_layers) == M


    # reconstruct original parameterization
    for i in xrange(M):

        theta_ = numpy.array(cvxmod.value(thetas[i]))
        z_layer = active_layers[i]
        ellipse_stack[z_layer] = conic_to_ellipse(theta_)
        ellipse_stack[z_layer].cz = z_layer

    return ellipse_stack



if __name__ == "__main__":

    import data_processing
    dx, dy, dz, di, v = data_processing.artificial_data()
    #fit = fit_ellipse_stack_scipy(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack_abs(dx, dy, dz, di)
    fit1 = fit_ellipse_stack_squared(dx, dy, dz, di)

