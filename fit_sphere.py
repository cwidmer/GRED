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
@summary: Module that focuses on the simpler special case of fitting a stack of circles

"""

from collections import namedtuple

import scipy.optimize
import numpy
import util
import sympy

from util import Ellipse

loss = None


def fitting_obj_stack(param, dx, dy, dz, di):
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

    obj = 0

    gradient_c = [0.0, 0.0]
    gradient_r = [0.0]*(len(radii))

    #loss = Loss("eucledian_squared")
    #loss = Loss("algebraic_squared")
    #loss = Loss("algebraic_squared")
    
    for idx in range(len(dx)):

        x = dx[idx]
        y = dy[idx]
        z = dz[idx]

        r = radii[z]

        obj += loss.get_obj(x, y, cx, cy, r)
        gradient_c[0] += loss.get_grad("cx", x, y, cx, cy, r)
        gradient_c[1] += loss.get_grad("cy", x, y, cx, cy, r)
        gradient_r[z] += loss.get_grad("r", x, y, cx, cy, r)


    # smoothness regularizer
    for idx in xrange(len(radii)-1):
        obj += (radii[idx] - radii[idx+1])**2

        # compute gradient
        if idx == 0:
            gradient_r[idx] += 2*radii[0] - 2*radii[1]
        else:
            gradient_r[idx] += 4*radii[idx] - 2*radii[idx-1] - 2*radii[idx+1]

    # last entry of gradient
    gradient_r[-1] += 2*radii[-1] - 2*radii[-2]

    # L1-regularize large radii
    for idx, r in enumerate(radii):
        obj += r
        gradient_r[idx] += 1

    # build final gradient
    gradient = gradient_c + gradient_r
 

    return obj, gradient


def check_gradient():
    """
    sanity check for gradient that compares the analytical gradient 
    to one computed numerically by finite differences
    """

    n = 10
    num_z = 2


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

    x0 = [3.0]*((max(z)+1) + 2)

    print "len(x0) = %i" % len(x0)

    # wrap function
    def func(param, x, y, z, i):
        return fitting_obj_stack(param, x, y, z, i)[0]

    def func_prime(param, x, y, z, i):
        return fitting_obj_stack(param, x, y, z, i)[1]


    print scipy.optimize.check_grad(func, func_prime, x0, x, y, z, i)


def fit_sphere_stack(dx, dy, dz, di):
    """
    fit ellipoid beased on data
    """

    #TODO think about what to do if there is not data on every layer
    #solution: regularize radii to zero, center to previous

    #global x,y,z,i
    x = numpy.array(dx)
    y = numpy.array(dy)
    z = numpy.array(dz)
    i = numpy.array(di)

    num_layers = max(z)+1
    print "number of active layers", num_layers
    print "num data points: %i" % (len(x))

    initial_radius = 5.0

    x0 = numpy.ones(num_layers+2)*initial_radius
    x0[0] = numpy.average(x)
    x0[1] = numpy.average(y)

    #x_opt = scipy.optimize.fmin(fitting_obj, x0)
    epsilon = 0.5

    # contrain all variables to be positive
    bounds = [(0,None) for idx in range(num_layers+2)]

    assert len(bounds) == len(x0)

    #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
    #x_opt = scipy.optimize.fmin(fitting_obj_sphere_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    #x_opt = scipy.optimize.fmin(fitting_obj_stack, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    #x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fit_sphere_c.fitting_obj_stack_cython, x0, bounds=bounds, approx_grad=True, messages=5, args=(x,y,z,i), epsilon=epsilon)
    #x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack, x0, bounds=bounds, approx_grad=True, messages=5, args=(x,y,z,i), epsilon=epsilon)
    x_opt, nfeval, rc = scipy.optimize.fmin_tnc(fitting_obj_stack, x0, bounds=bounds, approx_grad=False, messages=5, args=(x,y,z,i), epsilon=epsilon)

    #x_opt, nfeval, rc = scipy.optimize.fmin_l_bfgs_b(fitting_obj, x0, bounds=bounds, approx_grad=True, iprint=5)
    #x_opt = scipy.optimize.fmin(fitting_obj_sample, x0, xtol=epsilon, ftol=epsilon, disp=True, full_output=True)[0]
    
    ellipse_stack = []
    cx, cy = x_opt[0], x_opt[1]

    for z, radius in enumerate(x_opt[2:]):
        ellipse_stack.append(Ellipse(cx, cy, z, radius, radius, 0))

    return ellipse_stack


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
        self.cx = cx = sympy.Symbol("cx")
        self.cy = cy = sympy.Symbol("cy")
        self.r = r = sympy.Symbol("r")
        
        if loss_type == "eucledian_squared":
            self.fun = (sympy.sqrt((x-cx)**2 + (y-cy)**2) - r)**2

        if loss_type == "eucledian_abs":
            self.fun = sympy.sqrt((sympy.sqrt((x-cx)**2 + (y-cy)**2) - r)**2 + 0.001)

        if loss_type == "algebraic_squared":
            self.fun = ((x-cx)**2 + (y-cy)**2 - r)**2


        #TODO replace x**2 with x*x
        self.fun = self.fun.expand(deep=True)
        sympy.pprint(self.fun)

        self.d_cx = self.fun.diff(cx).expand(deep=True)
        self.d_cy = self.fun.diff(cy).expand(deep=True)
        self.d_r = self.fun.diff(r).expand(deep=True)


        # generate native code
        native_lang = "C"

        # generate native code
        if native_lang == "fortran":
            self.c_fun = autowrap(self.fun, language="F95", backend="f2py")
            self.c_d_cx = autowrap(self.d_cx)
            self.c_d_cy = autowrap(self.d_cy)
            self.c_d_r = autowrap(self.d_r)
        else:
            self.c_fun = autowrap(self.fun, language="C", backend="Cython", tempdir=".")
            self.c_d_cx = autowrap(self.d_cx, language="C", backend="Cython", tempdir=".")
            self.c_d_cy = autowrap(self.d_cy, language="C", backend="Cython", tempdir=".")
            self.c_d_r = autowrap(self.d_r, language="C", backend="Cython", tempdir=".")

        self.grads = {"cx": self.d_cx, "cy": self.d_cy, "r": self.d_r}
        self.c_grads = {"cx": self.c_d_cx, "cy": self.c_d_cy, "r": self.c_d_r}


    def get_obj(self, x, y, cx, cy, r):
        """
        eval objective at point
        """

        #obj = float(self.fun.evalf(subs = {self.x: x, self.y: y, self.cx: cx, self.cy: cy, self.r: r}))
        c_obj = float(self.c_fun(cx, cy, r, x, y))

        #print obj, c_obj
        #assert numpy.abs(obj - c_obj < 0.000001)

        #TODO figure out why the gradient is screwed up when using c_obj instead!

        return c_obj


    def get_grad(self, var_name, x, y, cx, cy, r):
        """
        eval gradient for variable at point
        """
        
        #grad = float(self.grads[var_name].evalf(subs = {self.x: x, self.y: y, self.cx: cx, self.cy: cy, self.r: r}))
        grad = float(self.c_grads[var_name](cx, cy, r, x, y))
        return grad


    def generate_c_code(self):
        """
        generate C-code to use with cython
        """

        sympy.printing.ccode(self.fun)


#loss = Loss("algebraic_squared")
#loss = Loss("eucledian_abs")
loss = Loss("eucledian_squared")

if __name__ == "__main__":

    check_gradient()
