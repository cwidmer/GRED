#!/usr/bin/env python2.5
#
# Written (W) 2011-2013 Christian Widmer
# Copyright (C) 2011-2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@summary: Some utility functions for the project, like sampling from an ellipse or ellipsoid

"""

import random
import numpy
import pylab
import math


class Ellipse(object):
    """
    ellipse class defining some plotting and sampling procedures
    """

    def __init__(self, cx, cy, cz, rx, ry, alpha):
        """
        constructor defining fields consisting of 
        center coordinates, radii and rotation angle
        """

        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.rx = rx
        self.ry = ry
        self.alpha = alpha


    def __str__(self):
        """
        string representation
        """

        return "cx=%.2g, cy=%.2g, cz=%.2g, rx=%.2g, ry=%.2g, alpha=%.2g" % (self.cx, self.cy, self.cz, self.rx, self.ry, self.alpha)


    def to_vector(self):
        """
        return numpy vector
        """

        return [self.cx, self.cy, self.cz, self.rx, self.ry, self.alpha]


    def plot(self, num_points=100, style="bo"):
        """
        plot ellipse by sampling points
        """

        self.plot_noshow(num_points, style)
        pylab.show()
        

    def plot_noshow(self, num_points=100, style="bo", label=""):
        """
        plot ellipse by sampling points
        """

        dat_x, dat_y = self.sample_equidistant(num_points)
        
        pylab.plot(dat_x, dat_y, style, label=label)


    def sample_equidistant(self, num_points):
        """
        sample from ellipsoid with equal angular spacing
        """

        theta = numpy.linspace(0, 2 * numpy.pi, num_points + 1)
    
        return self.sample_given_spacing(theta)


    def sample_uniform(self, num_points):
        """
        sample from ellipsoid with uniformly distributed spacing
        """

        theta = [random.uniform(0, 2 * numpy.pi) for _ in xrange(num_points + 1)]
    
        return self.sample_given_spacing(theta)


    def sample_given_spacing(self, theta):
        """
        sample points from ellipse given set of angles

        center: xc, yc
        radii: xr, yr
        rotation: alpha

        X = Z + Q(ALPHA) * [A * cos(theta); B * sin(theta)]
        where Q(ALPHA) is the rotation matrix
        Q(ALPHA) = [cos(ALPHA), -sin(ALPHA); 
                    sin(ALPHA), cos(ALPHA)]
        """

        # set up points
        points = numpy.zeros((2, len(theta)))
        points[0, :] = self.rx * numpy.cos(theta)
        points[1, :] = self.ry * numpy.sin(theta)

        ## get initial rotation matrix
        rot = numpy.array( [[ numpy.cos(self.alpha), -numpy.sin(self.alpha) ], 
                            [ numpy.sin(self.alpha),  numpy.cos(self.alpha) ]])
        
        ## center
        center = numpy.zeros((2, 1))
        center[0] = self.cx
        center[1] = self.cy

        # perform rotation
        dat = numpy.dot(rot , points) + center

        return dat


def conic_to_ellipse(theta, use_rotation=False):
    """
    convert theta parameterization to Ellipse object
    """

    #TODO support rotation
    ## For clarity, fill in the quadratic form variables
    #A        = numpy.zeros((2,2))
    #A[0,0]   = theta[0]
    #A.ravel()[1:3] = theta[2]
    #A[1,1]   = theta[1]
    #bv       = theta[3:5]
    #c        = theta[5]

    A              = numpy.zeros((2,2))
    A[0,0]         = theta[0]
    A.ravel()[1:3] = 0 #theta[2]
    A[1,1]         = theta[1]
    bv             = theta[2:4]
    c              = theta[4]
    
    ## find parameters
    z, a, b, alpha = conic2parametric(A, bv, c)

    return Ellipse(float(z[0]), float(z[1]), 0, float(a), float(b), float(alpha))


def conic2parametric(A, bv, c):
    """
    convert conic parameterization to standard ellipse parameterization
    """

    ## Diagonalise A - find Q, D such at A = Q' * D * Q
    D, Q = numpy.linalg.eig(A)
    Q = Q.T
    
    ## If the determinant < 0, it's not an ellipse
    if numpy.prod(D) <= 0:
        raise RuntimeError, 'fitellipse:NotEllipse Linear fit did not produce an ellipse'
    
    ## We have b_h' = 2 * t' * A + b'
    t = -0.5 * numpy.linalg.solve(A, bv)
    
    c_h = numpy.dot( numpy.dot( t.T, A ), t ) + numpy.dot( bv.T, t ) + c
    
    z = t
    a = numpy.sqrt(-c_h / D[0])
    b = numpy.sqrt(-c_h / D[1])
    alpha = math.atan2(Q[0,1], Q[0,0])
    
    return z, a, b, alpha


def ellipsoid(xc, yc, zc, xr, yr, zr, n=200):
    """
    sample points from ellipsoid

    note: does not support rotation
    """

    pi = numpy.pi
    theta = numpy.linspace (0, 2 * pi, n + 1);
    phi = numpy.linspace (-pi / 2, pi / 2, n + 1);
    [theta, phi] = numpy.meshgrid (theta, phi);

    lx = xr * numpy.cos(phi) * numpy.cos(theta) + xc;
    ly = yr * numpy.cos(phi) * numpy.sin(theta) + yc;
    lz = zr * numpy.sin(phi) + zc;

    return lx.flatten(), ly.flatten(), lz.flatten()


def plot_point_cloud(x,y,z):
    """
    plot 3d point cloud
    """

    import mpl_toolkits.mplot3d.axes3d as p3
    fig = pylab.figure()
    ax = p3.Axes3D(fig)
    ax.scatter(x,y,z)

    pylab.show()


def plot_ellipse(cx, cy, cz, rx, ry, alpha, figure):
    """
    plot ellipse stack
    """


    from mayavi import mlab

    n = 50


    dat = Ellipse(cx, cy, rx, ry, alpha).sample_uniform(n)
    #dat = ellipse(e.cx, e.cy, e.rx, e.ry, e.alpha, n)
    
    dx = dat[0]
    dy = dat[1]
    dz = [cz]*(n+1)
    dv = [25]*(n+1)

    return mlab.plot3d(dx,dy,dz,dv,
                        #mode='2dcircle',
                        #mode='point',
                        color=(0, 0, 0),
                        #scale_factor=100*max(self.data.shape),
                        figure=figure,
                        line_width=20,
                        tube_radius=None
                    )


def main():
    """
    main
    """

    ellipse = Ellipse(1, 1, 0, 2, 3, 0)
    ellipse.plot()


if __name__ == "__main__":
    main()

