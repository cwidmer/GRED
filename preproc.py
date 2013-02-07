#!/usr/bin/env python2.5
#
# Written (W) 2012 Christian Widmer
# Copyright (C) 2012 Max-Planck-Society

"""
@author: Christian Widmer
@summary: Preprocess larger volumes containing several cells, segment nuclei

"""

import os
import Image

import vigra.filters
from vigra.impex import writeVolume, writeImage
#, writeSlices

import numpy
import pylab
from matplotlib.patches import Polygon


def load_data3D(target):
    """
    load stack of tiffs
    """

    tif_dir = "data/whole_volume/20091026_SK570_590_4.5um_10_R3D_CAL_01_D3D" #data/whole_volume/20091026_SK570_590_4.5um_13_R3D_CAL_01_D3D/"
    tiffs = [os.path.join(str(tif_dir), f) for f in os.listdir(tif_dir) if f.endswith(".tif") and f.find(target) != -1]
    tiffs.sort()

    # grab dimensions
    dim_x, dim_y = vigra.impex.readImage(tiffs[0]).shape
    dim_z = len(tiffs)

    volume = vigra.ScalarVolume((dim_x, dim_y, dim_z))

    for (idx, tiff) in enumerate(tiffs):

        data = vigra.impex.readImage(tiff)
        volume[:,:,idx] = data

    print "loaded volume shape", volume.shape

    return volume


def plot_image_show(data, title=""):

    return ""

    #TODO implement for volumes
    mid_z = data.shape[2] / 2
    dat2d = data[:,:,mid_z]

    print "new shape", dat2d.shape

    pylab.figure()

    plot_image(dat2d, title)
    pylab.title(title)

    pylab.show()


def plot_image(data, title="", alpha=1.0):
    """
    plot 2d image (work around numpy-vigra compatability problem)
    """

    tmp_array = numpy.zeros(data.shape)

    for i in xrange(data.shape[0]):
        for j in xrange(data.shape[1]):
            tmp_array[i,j] = data[i,j]

    print "img shape", tmp_array.shape
    pylab.imshow(tmp_array, interpolation="nearest", alpha=alpha)


def extract():
    """
    This function localizes blob-like object using multi-scale hessian
    aggregation. The algorithm has been described in 
    [*} Xinghua Lou, X. Lou, U. Koethe, J. Wittbrodt, and F. A. Hamprecht. 
    Learning to Segment Dense Cell Nuclei with Shape Prior. In The 25th 
    IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2012), 2012.

    Adapted to python by Christian Widmer
    """

    # get data
    data = load_data3D("w617")
    data_green = load_data3D("w528")

    #scales = numpy.linspace(2.2, 6, 3)
    #scales = numpy.array([6.0])
    scales = numpy.array([5.0])
    #scales = numpy.array([3.0])
    closing = True
    opening = False
    window = 3
    thresholds = -0.01*numpy.array([1, 2, 3])
    conn = 0
    margin = 0
    verbose = True
    #ratios = numpy.array([1, 1, 0.25])
    #sigmas = numpy.array([1, 2, 4, 8])

    #seeds = arg(varargin, mstring('init'), true(size(data)))
    seeds = numpy.ones(data.shape)
    plot_image_show(data, title="raw image")


    for scale in scales:

        # sigma
        if verbose:
            print 'analyzing at sigma = %s' % (scale)

        # smooth image at this scale

        tmp = vigra.filters.gaussianSmoothing(data, (scale, scale, scale*0.5))
        plot_image_show(tmp, title="smoothed Gaussian")

        # compute eigenvalues
        #eigenValues = vigra.filters.eigenValueOfHessianMatrix(tmp, sigma, 0.9 * numpy.array([1, 1, 1]), mask, seeds)
        #hessian = vigra.filters.hessianOfGaussianEigenvalues(tmp, tmp_sigma)#, sigma_d=0.0, step_size=1.0, window_size=0.0, roi=None)

        hessian = vigra.filters.hessianOfGaussian3D(tmp, 0.4) #, tmp_sigma)#, sigma_d=0.0, step_size=1.0, window_size=0.0, roi=None)
        print "hessian.shape", hessian.shape
        plot_image_show(hessian[:,:,:,3], title="hessian")

        ev = vigra.filters.tensorEigenvalues(hessian)
        plot_image_show(ev[:,:,:,0], title="eigenvalue 0")
        plot_image_show(ev[:,:,:,1], title="eigenvalue 1")
        plot_image_show(ev[:,:,:,1], title="eigenvalue 2")

        print "ev.shape", ev.shape

        # combine eigenvalue indicators: xor
        if data.ndim == 3:
            seeds = numpy.logical_and(seeds, ev[:,:,:,0] < thresholds[0])
            seeds = numpy.logical_and(seeds, ev[:,:,:,1] < thresholds[1])
            seeds = numpy.logical_and(seeds, ev[:,:,:,2] < thresholds[2])
        elif data.ndim == 2:
            seeds = numpy.logical_and(seeds, ev[:,:,0] < thresholds[0])
            seeds = numpy.logical_and(seeds, ev[:,:,1] < thresholds[1])

        
        plot_image_show(seeds, title="seeds")

        seed_img = numpy.array(seeds, dtype=numpy.uint8)

        closed = vigra.filters.discClosing(seed_img, 3)
        plot_image_show(closed, title="closed seed")

        dilated = vigra.filters.discDilation(closed, 3)
        plot_image_show(dilated, title="dilated seed")


        print "dilated.shape", dilated.shape

        # heart piece

        # 
        detect_boxes(data, data_green, dilated)


        import ipdb
        ipdb.set_trace()

        pylab.figure()
        plot_image(data[:,:,8], title="seg vs real", alpha=0.5)
        plot_image(dilated[:,:,8], title="seg vs real", alpha=0.5)
        pylab.show()

        return ""


    #igra.filters.discClosing()
    #http//hci.iwr.uni-heidelberg.de/vigra/doc/vigranumpy/index.html?highlight=dilate

    #dilation operator afterwards

    #vigra.analysis.labelVolume()
    #vigra.analysis.labelImage()


def detect_boxes(data_red, data_grn, vol):
    """
    routine to automatically detect boxes in segmented image
    """

    labels_numpy = numpy.array(vol, dtype=numpy.uint8)

    labels = vigra.analysis.labelVolume(labels_numpy)
    plot_image_show(labels, title="labels")


    # determine unique cell labels
    unique = range(2, numpy.max(labels))

    num_kept = 0

    for idx in unique:

        # set up target volumes
        tvol_red = cut_region(data_red, labels, idx)
        tvol_grn = cut_box(data_grn, labels, idx)

        # do some basic filtering
        if tvol_red.shape[0] > 10 and tvol_red.shape[1] > 10 and tvol_red.shape[2] > 10:
            num_kept += 1

            out_dir = "test/cell_idx_%03d" % (idx)
            write_volume(tvol_red, out_dir, "w617")
            write_volume(tvol_grn, out_dir, "w528")


    print "keeping %i nuclei" % (num_kept)



def cut_region(raw, labels, target_label):
    """
    cut out labeled voxels only
    """

    a = numpy.array(labels)
    px, py, pz = numpy.where(a == target_label)

    assert len(pz) == len(py) == len(px)

    # determine dimensions
    d_x = int(max(px) - min(px)) + 1
    d_y = int(max(py) - min(py)) + 1
    d_z = int(max(pz) - min(pz)) + 1

    # set up target volume
    vol_new = numpy.zeros((d_x, d_y, d_z))

    # min on each axis
    mx = min(px)
    my = min(py)
    mz = min(pz)

    # copy box
    for i in xrange(d_x):
        for j in xrange(d_y):
            for k in xrange(d_z):

                tx = mx + i
                ty = my + j
                tz = mz + k

                vol_new[i, j, k] = raw[tx, ty, tz]

    return vol_new


def cut_box(raw, labels, target_label):
    """
    cut out entire box
    """

    a = numpy.array(labels)
    px, py, pz = numpy.where(a == target_label)

    assert len(pz) == len(py) == len(px)

    # determine dimensions
    d_x = int(max(px) - min(px)) + 1
    d_y = int(max(py) - min(py)) + 1
    d_z = int(max(pz) - min(pz)) + 1

    # set up target volume
    vol_new = numpy.zeros((d_x, d_y, d_z))

    # min on each axis
    mx = min(px)
    my = min(py)
    mz = min(pz)

    # copy box
    for i in xrange(d_x):
        for j in xrange(d_y):
            for k in xrange(d_z):

                tx = mx + i
                ty = my + j
                tz = mz + k

                vol_new[i, j, k] = raw[tx, ty, tz]

    return vol_new


def write_volume(vol, out_dir, color):
    """
    write volume using python's Image lib
    """

    #import pylab
    #pylab.hist(vol.flatten(), bins=100)
    #pylab.show()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for z in xrange(vol.shape[2]):
        img = vol[:,:,z]

        fn_base = "%s/layer_%02d_%s.tif" % (out_dir, z, color)
        fn_ext = "TIFF"

        #skimage.io.imsave(fn_base, img)
        im = Image.fromarray(img)

        #print im.size
        #print im.mode
        #table=[ i/256 for i in range(65536) ]
        #im2 = im.point(table,'L')

        #im.convert('L').save(fn_base, fn_ext)
        im.save(fn_base, fn_ext)


def write_volume_vigra(vol, out_dir, color):
    """
    write volume using vigra (codec seems to be missing)
    """

    fn_base = "%s/layer_%s.tif" % (out_dir, color)
    fn_ext = "tif"
    #writeVolume(tvol_numpy, fn_base, fn_ext, dtype = '', compression = '')
    writeVolume(vol, fn_base, fn_ext)


if __name__ == "__main__":
    extract()

if __name__ == "pyreport.main":
    extract()

