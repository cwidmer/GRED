#!/usr/bin/env python2.5
#
# Written (W) 2011-2012 Christian Widmer
# Copyright (C) 2011-2012 Max-Planck-Society

"""
@author: Christian Widmer

@summary: module contains autowrapped code, which is executed once on import
"""

import sympy
from sympy.utilities.autowrap import autowrap

def setup_loss(loss_type, native_lang = "C", verbose=False):
    """
    set up symbolic derivations
    """

    print "setting up autowrapped loss", loss_type

    x = x = sympy.Symbol("x")
    y = y = sympy.Symbol("y")
    z = z = sympy.Symbol("z")
    cx = cx = sympy.Symbol("cx")
    cy = cy = sympy.Symbol("cy")
    wx = wx = sympy.Symbol("wx")
    wy = wy = sympy.Symbol("wy")
    rx = rx = sympy.Symbol("rx")
    ry = ry = sympy.Symbol("ry")

    if loss_type == "algebraic_squared":
        fun = ((x-(cx + z*wx))**2/(rx*rx) + (y-(cy + z*wy))**2/(ry*ry) - 1)**2

    if loss_type == "algebraic_abs":
        fun = sympy.sqrt(((x-(cx + z*wx))**2/(rx*rx) + (y-(cy + z*wy))**2/(ry*ry) - 1)**2 + 0.01)

    #TODO replace x**2 with x*x
    fun = fun.expand(deep=True)

    if verbose:
        sympy.pprint(fun)

    d_cx = fun.diff(cx).expand(deep=True)
    d_cy = fun.diff(cy).expand(deep=True)
    d_wx = fun.diff(wx).expand(deep=True)
    d_wy = fun.diff(wy).expand(deep=True)
    d_rx = fun.diff(rx).expand(deep=True)
    d_ry = fun.diff(ry).expand(deep=True)

    if native_lang == "fortran":
        c_fun = autowrap(fun, language="F95", backend="f2py")
        c_d_cx = autowrap(d_cx)
        c_d_cy = autowrap(d_cy)
        w_d_wx = autowrap(d_wx)
        w_d_wy = autowrap(d_wy)
        c_d_rx = autowrap(d_rx)
        c_d_ry = autowrap(d_ry)
    else:
        c_fun = autowrap(fun, language="C", backend="Cython", tempdir=".")
        c_d_cx = autowrap(d_cx, language="C", backend="Cython", tempdir=".")
        c_d_cy = autowrap(d_cy, language="C", backend="Cython", tempdir=".")
        c_d_wx = autowrap(d_wx, language="C", backend="Cython", tempdir=".")
        c_d_wy = autowrap(d_wy, language="C", backend="Cython", tempdir=".")
        c_d_rx = autowrap(d_rx, language="C", backend="Cython", tempdir=".")
        c_d_ry = autowrap(d_ry, language="C", backend="Cython", tempdir=".")

    grads = {"cx": d_cx, "cy": d_cy, "wx": d_wx, "wy": d_wy, "rx": d_rx, "ry": d_ry}
    c_grads = {"cx": c_d_cx, "cy": c_d_cy, "wx": c_d_wx, "wy": c_d_wy, "rx": c_d_rx, "ry": c_d_ry}

    print "done setting up autowrapped loss"

    return c_fun, c_grads

losses = {}
losses["algebraic_squared"] = setup_loss("algebraic_squared")
#losses["algebraic_abs"] = setup_loss("algebraic_abs")


class Loss(object):
    """
    derive gradient for various loss functions using sympy
    """

    def __init__(self, loss_type):
        
        if loss_type == "algebraic_squared":
            self.c_fun = losses["algebraic_squared"][0]
            self.c_grads = losses["algebraic_squared"][1]

        elif loss_type == "algebraic_abs":
            self.c_fun = losses["algebraic_abs"][0]
            self.c_grads = losses["algebraic_abs"][1]

        else:
            print "WARNING: unknown loss function"

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

