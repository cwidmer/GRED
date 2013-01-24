#!/usr/bin/env python2.5
#
# Written (W) 2013 Christian Widmer
# Copyright (C) 2013 Max-Planck-Society, TU-Berlin, MSKCC

"""
@author: Christian Widmer
@created: 2013-01-10
@summary: Creating artifical data sets for publication

"""

import numpy
import random
import pylab
import util
import fit_ellipse_conic


def compare_ellipses(lhs, rhs):
    """
    report squared error wrt parameter vectors
    """
    
    v_lhs, v_rhs = lhs.to_vector(), rhs.to_vector()
    assert len(v_lhs) == len(v_rhs)

    error = 0

    for i in xrange(len(v_lhs)):
        error += numpy.square(v_lhs[i] - v_rhs[i])

    return error


def sample_ellipse_gaussian(ellipse, num_points, num_errors, plot=False):
    """
    sample from ellipse, apply gaussian noise
    """

    dat_x, dat_y = ellipse.sample_uniform(num_points)

    # add gaussian noise to original data
    for i in xrange(len(dat_x)):
        dat_x[i] += random.gauss(0, 0.2)
        dat_y[i] += random.gauss(0, 0.2)

    # create uniformly distributed errors
    err_x, err_y = numpy.zeros(num_errors), numpy.zeros(num_errors)

    for i in xrange(num_errors):
        err_x[i] = random.uniform(-3*ellipse.rx, 3*ellipse.rx)
        err_y[i] = random.uniform(-3*ellipse.rx, 3*ellipse.rx)

    # concat real and errorous data
    all_x = numpy.concatenate((dat_x, err_x))
    all_y = numpy.concatenate((dat_y, err_y))

    # fit
    ellipse_fit_squared = fit_ellipse_conic.fit_ellipse_squared(all_x, all_y)
    ellipse_fit_eps = fit_ellipse_conic.fit_ellipse_eps_insensitive(all_x, all_y)

    # visualize
    if plot:
        ellipse.plot_noshow(style="g-", label="ground truth")
        pylab.plot(dat_x, dat_y, "go", label="sampled data")
        pylab.plot(err_x, err_y, "mo", label="uniform noise")
        ellipse_fit_squared.plot_noshow(style="r-", label="fit squared loss")
        ellipse_fit_eps.plot_noshow(style="b-", label="robust loss")

        #pylab.legend(loc="upper left")
        pylab.legend(loc="best")
        pylab.xlim((-3,3))
        pylab.ylim((-3,3))

        pylab.grid()
        pylab.show()

    error_squared = compare_ellipses(ellipse, ellipse_fit_squared)
    error_eps = compare_ellipses(ellipse, ellipse_fit_eps)

    print "fit squared", ellipse_fit_squared, "error", error_squared
    print "fit eps", ellipse_fit_eps, "error", error_eps

    return error_squared, error_eps



def run_robustness_experiment(num_errors, num_repeats):
    """
    comparsed sensitivity to outliers for squared and eps-sensitive loss
    """


    ellipse = util.Ellipse(0, 0, 0, 1, 1.5, 0)

    print "initial", ellipse

    errors_squared = []
    errors_eps = []

    for i in xrange(num_errors):

        tmp_squared = []
        tmp_eps = []

        for _ in xrange(num_repeats):

            try:
                err_squared, err_eps = sample_ellipse_gaussian(ellipse, 40, i)

                tmp_squared.append(err_squared)
                tmp_eps.append(err_eps)
            except Exception, detail:
                print detail
        

        errors_squared.append(numpy.median(tmp_squared))
        errors_eps.append(numpy.median(tmp_eps))

    pylab.title("robustness analysis")
    pylab.xlabel("uniform noise")
    pylab.ylabel("error")

    pylab.grid()
    pylab.plot(errors_squared, "-r", label="squared loss")
    pylab.plot(errors_eps, "-b", label="robust loss")
    pylab.legend(loc="upper left")

    pylab.show()


def fit_circle_houghtransform():
    """
    use hough transform to fit circle:
    http://www.janeriksolem.net/2012/08/reading-gauges-detecting-lines-and.html
    """

    #prefix="/home/cwidmer/Documents/phd/projects/cell_fitting/data/data/20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9"
    #fn = prefix + "/" + "20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9_w617_z08.tif"
    #im = np.array(dp.image2array(Image.open(fn)), dtype=np.uint8)

    import cv2
    import Image
    import numpy as np
    import data_processing as dp

    im = np.array(dp.image2array(Image.open(fn)), dtype=np.uint8)
    m,n = im.shape
    circles = cv2.HoughCircles(im, cv2.cv.CV_HOUGH_GRADIENT, 2, 10, np.array([]), 20, 60, m/10)[0]
    c = circles[0]
    draw_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.circle(draw_im, (c[0],c[1]), c[2], (0,255,0), 2)
    cv2.imshow("circles",im)
    cv2.waitKey()
    cv2.imwrite("res.jpg",draw_im)



def main():
    """
    main function
    """

    num_errors = 20
    num_repeats = 50
    run_robustness_experiment(num_errors, num_repeats)

    ellipse = util.Ellipse(0, 0, 0, 1, 1.5, 0)
    sample_ellipse_gaussian(ellipse, 40, 2, plot=True)

    ellipse = util.Ellipse(0, 0, 0, 1, 1.5, 0)
    sample_ellipse_gaussian(ellipse, 40, 10, plot=True)

    ellipse = util.Ellipse(0, 0, 0, 1, 1.5, 0)
    sample_ellipse_gaussian(ellipse, 40, 20, plot=True)



if __name__ == "__main__":
    main()


