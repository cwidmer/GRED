#!/usr/bin/env python2.5
#
# Written (W) 2011-2013 Christian Widmer
# Copyright (C) 2011-2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@summary: Procedures for fitting ellipses using 
          conic parameterization

"""


import numpy
import cvxmod

from util import conic_to_ellipse



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

