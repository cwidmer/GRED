#!/usr/bin/env python2.5
#
# Written (W) 2013 Christian Widmer
# Copyright (C) 2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@created: 2013-01-10
@summary: Creating artifical data sets for publication

"""

import random
import pylab
import util
import fit_ellipse_conic


def sample_ellipse_gaussian(ellipse, num_points):
    """
    sample from ellipse, apply gaussian noise
    """

    dat_x, dat_y = ellipse.sample_uniform(num_points)


    for i in xrange(len(dat_x)):
        dat_x[i] += random.gauss(0, 0.4)
        dat_y[i] += random.gauss(0, 0.4)

    pylab.plot(dat_x, dat_y, "bo")

    ellipse = fit_ellipse_conic.fit_ellipse_squared(dat_x, dat_y)
    ellipse.plot_noshow(style="r-")
    pylab.show()

    print "fit", ellipse


def main():
    """
    main function
    """

    ellipse = util.Ellipse(1, 1, 0, 2, 3.5, 0)
    ellipse.plot_noshow(style="b-")

    print "initial", ellipse

    sample_ellipse_gaussian(ellipse, 10)


if __name__ == "__main__":
    main()

