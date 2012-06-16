#!/usr/bin/env python2.5
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Christian Widmer
# Copyright (C) 2011 Max-Planck-Society

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
    ellipse_stack = []
    for idx in range(max(dz)):
        ellipse_stack.append(Ellipse(0, 0, idx, 1, 1, 0))
    

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



def fitting_obj_stack(param, x, y, z, i):
    """
    computes residuals based on distance from ellipsoid
    
    can be used with different loss-functions on residual
    """


    # centers
    cx = param[0]
    cy = param[1]

    #num_layers = len(set(z))
    #assert len(param) == num_layers+2

    radii = param[2:]

    num_layers = len(radii) / 2
    radii_x = radii[0:num_layers]
    radii_y = radii[num_layers:]

    obj = 0


    for idx in range(len(x)):

        rx = radii_x[z[idx]]
        ry = radii_y[z[idx]] 
        residual = (cx - x[idx])**2 / (rx**2) + (cy - y[idx])**2 / (ry**2) - 1

        tmp = loss_functions.squared_loss(residual)
        #obj += loss_functions.eps_loss(residual, 2)*data_intensity[idx]
        #obj += loss_functions.abs_loss(residual)
        #tmp = loss_functions.eps_loss(residual, 1)
        #obj += loss_functions.eps_loss_asym(residual, 2, 1.0, 0.3)
        #print idx, residual

        #obj += tmp*i[idx]
        obj += tmp


    #gradient = gradient_c + gradient_rx + gradient_ry
    # smoothness regularizer
    #for idx in xrange(len(radii_x)-1):
    #    obj += (radii_x[idx] - radii_x[idx+1])**2

    # smoothness regularizer
    #for idx in xrange(len(radii_y)-1):
    #    obj += (radii_y[idx] - radii_y[idx+1])**2

    # L1-regularize large radii
    #for r in radii:
    #    obj += r

    return obj


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


def check_gradient():
    """
    sanity check for gradient that compares the analytical gradient 
    to one computed numerically by finite differences
    """

    n = 10
    num_z = 10


    x = []
    y = []
    z = []
    i = []

    for idx in range(num_z):
        dat = util.ellipse(1, 1, 1, 2, 0, n=n)
    
        x += list(dat[0])
        y += list(dat[1])
        z += [idx]*(n+1)
        i += [1.0]*(n+1)

    assert len(x) == len(y) == len(z) == len(i)

    x0 = [3.0]*((max(z)+1)*2 + 2 + 2)

    print "len(x0) = %i" % len(x0)

    loss = Loss("algebraic_squared")

    # wrap function
    def func(param, x, y, z, i):
        return fitting_obj_stack_gradient(param, x, y, z, i, loss)[0]

    def func_prime(param, x, y, z, i):
        return fitting_obj_stack_gradient(param, x, y, z, i, loss)[1]


    print scipy.optimize.check_grad(func, func_prime, x0, x, y, z, i)




def fit_ellipse_stack_scipy(dx, dy, dz, di, loss_type = "algebraic_abs"):
    """
    fit ellipoid based on scipy optimize
    """

    #TODO think about what to do if there is not data on every layer
    #solution: regularize radii to zero, center to previous

    #global x,y,z,i
    x = numpy.array(dx)
    y = numpy.array(dy)
    z = numpy.array(dz)
    i = numpy.array(di)

    min_z = min(z)
    num_layers = max(z) - min_z + 1
    print "number of active layers", num_layers
    print "num data points: %i" % (len(x))

    initial_radius = 3.0

    num_parameters = 2 + 2 + num_layers*2 # center + w-vector + radii
    x0 = numpy.ones(num_parameters)*initial_radius
    x0[0] = numpy.average(x)
    x0[1] = numpy.average(y)
    x0[2] = 0
    x0[3] = 0

    #x_opt = scipy.optimize.fmin(fitting_obj, x0)
    epsilon = 0.1

    # contrain all variables to be positive
    bounds = [(0,None) for idx in range(num_parameters)]
    bounds[2] = (None, None) # no positivity for w0
    bounds[3] = (None, None) # no positivity for w1

    assert len(bounds) == len(x0)

    print "fitting ellipse stack with loss", loss_type
    loss = Loss(loss_type)

    #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
    #x_opt = scipy.optimize.fmin(fitting_obj_sphere_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    #x_opt = scipy.optimize.fmin(fitting_obj_stack, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    #x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack, x0, bounds=bounds, approx_grad=True, messages=5, args=(x,y,z,i), epsilon=epsilon)
    #x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack_gradient, x0, bounds=bounds, messages=5, args=(x,y,z,i), epsilon=epsilon)
    x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack_gradient, x0, bounds=bounds, messages=5, args=(x,y,z,i,loss), epsilon=epsilon)

    #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
    #x_opt = scipy.optimize.fmin(fitting_obj_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    
    ellipse_stack = {}
    cx, cy = x_opt[0], x_opt[1]
    wx, wy = x_opt[2], x_opt[3]

    print "cx, cy, wx, wy:", cx, cy, wx, wy

    radii = x_opt[4:]
    radii_x = radii[0:num_layers]
    radii_y = radii[num_layers:]
   
    assert len(radii_x) == len(radii_y) == num_layers

    # compile return datastructure
    for r_idx in xrange(num_layers):
        z_idx = r_idx + min_z
        tmp_cx = cx + r_idx*wx
        tmp_cy = cy + r_idx*wy
        ellipse_stack[z_idx] = Ellipse(tmp_cx, tmp_cy, z_idx, radii_x[r_idx], radii_y[r_idx], 0)

    return ellipse_stack




def fit_ellipse_eps_insensitive(x, y):
    """
    fit ellipoid using epsilon-insensitive loss
    """
    import cvxmod

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
    objective = cvxmod.sum(square(X*theta))
    
    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))
    p.constr.append(theta[0] + theta[1] == 1)
    
    
    ###### set values
    X.value = dat
    #solver = "mosek" 
    #p.solve(lpsolver=solver)
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
        dat[dz[idx]].append( [dx[idx], dy[idx]] )

    # init ret
    ellipse_stack = []
    for idx in range(max(dz)):
        ellipse_stack.append(Ellipse(0, 0, idx, 1, 1, 0))
    

    #TODO restrict to two
    M = len(dat.keys())
    D = 6

    X_matrix = []
    thetas = []

    for z in dat.keys():

        print "muh", dat[z]
        x = numpy.array(dat[z])[:,0]
        y = numpy.array(dat[z])[:,1]
        N = len(x)
        d = numpy.zeros((N, D))
        d[:,0] = x*x
        d[:,1] = y*y
        d[:,2] = x*y
        d[:,3] = x
        d[:,4] = y
        d[:,5] = numpy.ones(N)
    

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
        objective += cvxmod.sum(square(X*thetas[i]))

    # add smoothness regularization
    for i in xrange(M-1):
        objective += cvxmod.sum(square(thetas[i] - thetas[i+1]))

    print objective

    # create problem                                    
    p = cvxmod.problem(cvxmod.minimize(objective))

    # add constraints
    for i in xrange(M-1):
        p.constr.append(thetas[i][0] + thetas[i][1] == 1)
    
    
    ###### set values
    p.solve()
    

    print p
    
    return None

    w = numpy.array(cvxmod.value(theta))
    
    #print weights
    
    cvxmod.printval(theta)


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



class Loss(object):
    """
    derive gradient for various loss functions using sympy
    """

    def __init__(self, loss_type):
        """
        set up symbolic derivations
        """

        from sympy.utilities.autowrap import autowrap

        self.x = x = sympy.Symbol("x")
        self.y = y = sympy.Symbol("y")
        self.z = z = sympy.Symbol("z")
        self.cx = cx = sympy.Symbol("cx")
        self.cy = cy = sympy.Symbol("cy")
        self.wx = wx = sympy.Symbol("wx")
        self.wy = wy = sympy.Symbol("wy")
        self.rx = rx = sympy.Symbol("rx")
        self.ry = ry = sympy.Symbol("ry")
        

        if loss_type == "algebraic_squared":
            self.fun = ((x-(cx + z*wx))**2/(rx*rx) + (y-(cy + z*wy))**2/(ry*ry) - 1)**2

        if loss_type == "algebraic_abs":
            self.fun = sympy.sqrt(((x-(cx + z*wx))**2/(rx*rx) + (y-(cy + z*wy))**2/(ry*ry) - 1)**2 + 0.01)

        #TODO replace x**2 with x*x
        self.fun = self.fun.expand(deep=True)
        sympy.pprint(self.fun)

        self.d_cx = self.fun.diff(cx).expand(deep=True)
        self.d_cy = self.fun.diff(cy).expand(deep=True)
        self.d_wx = self.fun.diff(wx).expand(deep=True)
        self.d_wy = self.fun.diff(wy).expand(deep=True)
        self.d_rx = self.fun.diff(rx).expand(deep=True)
        self.d_ry = self.fun.diff(ry).expand(deep=True)


        # generate native code
        native_lang = "C"

        if native_lang == "fortran":
            self.c_fun = autowrap(self.fun, language="F95", backend="f2py")
            self.c_d_cx = autowrap(self.d_cx)
            self.c_d_cy = autowrap(self.d_cy)
            self.w_d_wx = autowrap(self.d_wx)
            self.w_d_wy = autowrap(self.d_wy)
            self.c_d_rx = autowrap(self.d_rx)
            self.c_d_ry = autowrap(self.d_ry)
        else:
            self.c_fun = autowrap(self.fun, language="C", backend="Cython", tempdir=".")
            self.c_d_cx = autowrap(self.d_cx, language="C", backend="Cython", tempdir=".")
            self.c_d_cy = autowrap(self.d_cy, language="C", backend="Cython", tempdir=".")
            self.c_d_wx = autowrap(self.d_wx, language="C", backend="Cython", tempdir=".")
            self.c_d_wy = autowrap(self.d_wy, language="C", backend="Cython", tempdir=".")
            self.c_d_rx = autowrap(self.d_rx, language="C", backend="Cython", tempdir=".")
            self.c_d_ry = autowrap(self.d_ry, language="C", backend="Cython", tempdir=".")

        self.grads = {"cx": self.d_cx, "cy": self.d_cy, "wx": self.d_wx, "wy": self.d_wy, "rx": self.d_rx, "ry": self.d_ry}
        self.c_grads = {"cx": self.c_d_cx, "cy": self.c_d_cy, "wx": self.c_d_wx, "wy": self.c_d_wy, "rx": self.c_d_rx, "ry": self.c_d_ry}


    def get_obj(self, x, y, z, cx, cy, wx, wy, rx, ry):
        """
        eval objective at point
        """

        c_obj = float(self.c_fun(cx, cy, rx, ry, wx, wy, x, y, z))

        return c_obj


    def get_grad(self, var_name, x, y, z, cx, cy, wx, wy, rx, ry):
        """
        eval gradient for variable at point
        """
        
        grad = float(self.c_grads[var_name](cx, cy, rx, ry, wx, wy, x, y, z))

        return grad

        
if __name__ == "__main__":

    print "checking gradient of abs loss"
    check_gradient()

    
    #fit = fit_ellipse_stack_scipy(dx, dy, dz, di)
    #fit1 = fit_ellipse_stack(dx, dy, dz, di)
    #fit1 = fit_ellipse_stacked_squared(dx, dy, dz, di)

