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

@summary: procedures to load and handle voxel data

"""

import numpy
import os
import Image


def get_data_example():
    """
    get some dataset
    """

    tif_dir = "data/data/20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9/"
    target = "w617"
    threshold = 255
    vol = load_tif(tif_dir, target)

    return vol
    
    #dx, dy, dz, di, vol = threshold_volume(vol, threshold)

    #return numpy.array(dx), numpy.array(dy), numpy.array(dz), numpy.array(di)


def image2array(im):
    """
    convert 16bit image to numpy array
    """

    arr = numpy.zeros(im.size)

    for x in xrange(im.size[0]):
        for y in xrange(im.size[1]):
            arr[x,y] = im.getpixel((x,y))

    return arr


def load_tif(tif_dir, target):
    """
    load data file
    """

    tiffs = [os.path.join(str(tif_dir), f) for f in os.listdir(tif_dir) if f.endswith(".tif") and f.find(target) != -1]
    tiffs.sort()

    img_layers = []

    for (idx, tiff) in enumerate(tiffs):

        # convert to numpy array
        myimg = image2array(Image.open(tiff))

        img_layers.append(myimg)


    # set up volume
    vol = numpy.zeros((myimg.shape[0], myimg.shape[1], len(img_layers)))
    for idx_x in range(myimg.shape[0]):
        for idx_y in range(myimg.shape[1]):
            for idx_z in range(len(img_layers)):

                vol[idx_x, idx_y, idx_z] = img_layers[idx_z][idx_x, idx_y]


    return vol


def threshold_volume(vol, threshold, std_cut):
    """
    given threshold, apply thresholding return a list of points 
    and thresholded volume

    std_cut: number of std wrt distance to center in which to cut
    """

    # perform thresholding
    b = vol.flatten()

    b[b < threshold] = 0
    b = b.reshape(vol.shape)

    # clear outliers
    vol = crop_data(b)

    # create points from volume
    d_x = []
    d_y = []
    d_z = []
    d_intensity = []

    for idx_x in range(vol.shape[0]):
        for idx_y in range(vol.shape[1]):
            for idx_z in range(vol.shape[2]):

                if vol[idx_x, idx_y, idx_z] > 0:
                    d_x.append(idx_x)
                    d_y.append(idx_y)
                    d_z.append(idx_z)
                    d_intensity.append(vol[idx_x, idx_y, idx_z])

    # cut points based on distance to center
    keep = cut_points(d_x, d_y, d_z, std_cut=std_cut)

    print "keep", keep[:10]

    # select keepers
    d_x = numpy.array(d_x)[keep]
    d_y = numpy.array(d_y)[keep]
    d_z = numpy.array(d_z)[keep]
    d_intensity = numpy.array(d_intensity)[keep]

    return d_x, d_y, d_z, d_intensity, vol


def cut_points(data_x, data_y, data_z, std_cut=3.0, debug=False):
    """
    cut points based on distance to center
    """

    assert len(data_x) == len(data_y) == len(data_z)
    dat = numpy.zeros((len(data_x), 3))
    dat[:,0] = data_x
    dat[:,1] = data_y
    dat[:,2] = data_z

    mean = dat.mean(axis=0)
    diff_vec = dat - mean
    distances = map(numpy.linalg.norm, diff_vec)

    norm_dist = (distances - numpy.mean(distances)) / numpy.std(distances)

    # cut everything that is more than std_cut std deviations from center
    keeper_idx = numpy.where(norm_dist <= std_cut)[0]

    assert len(keeper_idx) <= len(data_x)
    print "cut %f std deviations, keeping %i/%i points" % (std_cut, len(keeper_idx), len(data_x))

    if debug:
        import pylab
        pylab.hist(norm_dist, bins=100)
        pylab.show()

    return keeper_idx


def crop_data(vol):
    """
    DFS to crop data from the boundaries
    top and bottom boundaries are ignored
    """

    thres = 250

    num_x = vol.shape[0]
    num_y = vol.shape[1]
    num_z = vol.shape[2]

    
    # set up starting positions
    starts = []

    # front and back
    for i in range(num_x):
        for j in range(num_z):
            starts.append( (i, 0, j) )
            starts.append( (i, num_y-1, j) )

    # left and right
    for i in range(num_y):
        for j in range(num_z):
            starts.append( (0, i, j) )
            starts.append( (num_x-1, i, j) )

    # DFS
    seenpositions = set()
    currentpositions = set(starts)

    while currentpositions:
        nextpositions = set()
        for p in currentpositions:
            seenpositions.add(p)
            succ = possiblesuccessors(vol, p, thres)
            for np in succ:
                if np in seenpositions: continue
                nextpositions.add(np)

            currentpositions = nextpositions

    print "cropping %i (%i addional) voxels" % (len(seenpositions), len(seenpositions) - len(starts))

    # crop visited voxels
    for pos in seenpositions:
        vol[pos[0], pos[1], pos[2]] = 0.0

    return vol


def possiblesuccessors(vol, p, thres):
    """
    checks voxel in volume for possible successors
    """

    successors = []
    num_x = vol.shape[0]
    num_y = vol.shape[1]
    num_z = vol.shape[2]

    # left
    if p[0] > 0 and vol[p[0]-1, p[1], p[2]] >= thres:
        successors.append( (p[0]-1, p[1], p[2]) )

    # right
    if p[0] < num_x - 1 and vol[p[0]+1, p[1], p[2]] >= thres:
        successors.append( (p[0]+1, p[1], p[2]) )
             
    # below
    if p[1] > 0 and vol[p[0], p[1]-1, p[2]] >= thres:
        successors.append( (p[0], p[1]-1, p[2]) )

    # top
    if p[1] < num_y - 1 and vol[p[0], p[1]+1, p[2]] >= thres:
        successors.append( (p[0], p[1]+1, p[2]) )

    # up in z
    if p[2] > 0 and vol[p[0], p[1], p[2]-1] >= thres:
        successors.append( (p[0], p[1], p[2]-1) )

    # down in z
    if p[2] < num_z - 1 and vol[p[0], p[1], p[2]+1] >= thres:
        successors.append( (p[0], p[1], p[2]+1) )


    return successors



def generate_sphere_full():
    """
    sample data from sphere
    """
    
    num_voxels = 31
    c = (15.0, 15.0, 15.0)

    data_x = []
    data_y = []
    data_z = []
    data_intensity = []

    volume = numpy.zeros((num_voxels, num_voxels, num_voxels))

    for x in range(num_voxels):
        for y in range(num_voxels):
            for z in range(num_voxels):

                if numpy.sqrt((x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2) - 7.5 < 1.5:
                    data_x.append(x)
                    data_y.append(y)
                    data_z.append(z)
                    data_intensity.append(200.0)

                    volume[x,y,z] = 200.0


    return data_x, data_y, data_z, data_intensity, volume

def artificial_data():
    """
    sample data from sphere
    """
    
    num_voxels = 10
    c = (5.0, 5.0, 5.0)

    data_x = []
    data_y = []
    data_z = []
    data_intensity = []

    volume = numpy.zeros((num_voxels, num_voxels, num_voxels))

    for x in range(num_voxels):
        for y in range(num_voxels):
            for z in range(num_voxels):

                if numpy.abs(numpy.sqrt((x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2) - 5) < 1.5:
                    data_x.append(x)
                    data_y.append(y)
                    data_z.append(z)
                    data_intensity.append(200.0)

                    volume[x,y,z] = 200.0


    return data_x, data_y, data_z, data_intensity, volume


def generate_cube():
    """
    sample data from cube
    """
    
    num_voxels = 31

    data_x = []
    data_y = []
    data_z = []
    data_intensity = []

    volume = numpy.zeros((num_voxels, num_voxels, num_voxels))

    for x in range(num_voxels):
        for y in range(num_voxels):
            for z in range(num_voxels):

                if 5 < x < 10 and 5 < y < 10:
                    data_x.append(x)
                    data_y.append(y)
                    data_z.append(z)
                    data_intensity.append(200.0)

                    volume[x,y,z] = 200.0


    return data_x, data_y, data_z, data_intensity, volume


