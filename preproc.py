import vigra.filters
import numpy
import pylab


def load_data2D():
    #readImage(filename, dtype = 'FLOAT', index = 0, order='') -> Image
    #TODO read tiff files
    #tif_dir = "data/data/20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9/"
    #tif_file = "20091026_SK570_578_4.5um_1_R3D_CAL_01_D3D_CPY_Cut9_w617_z02.tif"a
    tif_dir = "data/whole_volume/20091026_SK570_590_4.5um_13_R3D_CAL_01_D3D/"
    tif_file = "20091026_SK570_590_4.5um_13_R3D_CAL_01_D3D_w617_z20.tif"
    data = vigra.impex.readImage(tif_dir + tif_file)

    print type(data)

    return data


def plot_image_show(data, title=""):

    pylab.figure()

    plot_image(data, title)
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

    #data = data_processing.get_data_example()
    data = load_data2D()

    #data = vigra.ScalarVolume((width, height, depth))
    #TODO: get 3D data into right format
   
    #scales = numpy.linspace(3, 15, 4)
    scales = numpy.array([5.0])
    closing = True
    opening = False
    window = 3
    thresholds = -0.05*numpy.array([1, 2, 3])
    conn = 0
    margin = 0
    verbose = True
    ratios = numpy.array([1, 1, 0.25])
    sigmas = numpy.array([1, 2, 4, 8])

    #seeds = arg(varargin, mstring('init'), true(size(data)))
    seeds = numpy.ones(data.shape)
    plot_image_show(data, title="raw image")


    for scale in scales:

        # sigma
        if verbose:
            print 'analyzing at sigma = %s' % (scale)

        # smooth image at this scale
        tmp = vigra.filters.gaussianSmoothing(data, scale)
        plot_image_show(tmp, title="smoothed Gaussian")

        # compute eigenvalues
        #eigenValues = vigra.filters.eigenValueOfHessianMatrix(tmp, sigma, 0.9 * numpy.array([1, 1, 1]), mask, seeds)
        #hessian = vigra.filters.hessianOfGaussianEigenvalues(tmp, tmp_sigma)#, sigma_d=0.0, step_size=1.0, window_size=0.0, roi=None)

        hessian = vigra.filters.hessianOfGaussian2D(tmp, 0.4) #, tmp_sigma)#, sigma_d=0.0, step_size=1.0, window_size=0.0, roi=None)
        plot_image_show(hessian, title="hessian")

        ev = vigra.filters.tensorEigenvalues(hessian)
        plot_image_show(ev[:,:,0], title="eigenvalue 0")
        plot_image_show(ev[:,:,1], title="eigenvalue 1")

        # combine eigenvalue indicators: xor
        if data.ndim == 3:
            seeds = numpy.logical_and(seeds, ev[:,:,0] < thresholds[0])
            seeds = numpy.logical_and(seeds, ev[:,:,1] < thresholds[1])
            seeds = numpy.logical_and(seeds, ev[:,:,2] < thresholds[2])
        elif data.ndim == 2:
            seeds = numpy.logical_and(seeds, ev[:,:,0] < thresholds[0])
            seeds = numpy.logical_and(seeds, ev[:,:,1] < thresholds[1])

        
        plot_image_show(seeds, title="seeds")

        #seed_img = vigra.ScalarImage(seeds)
        #seed_img = numpy.array(seeds, dtype=numpy.float32)
        seed_img = numpy.array(seeds, dtype=numpy.uint8)
        #vigra.ScalarVolume((30,30,30))

        closed = vigra.filters.discClosing(seed_img, 2)
        plot_image_show(closed, title="closed seed")

        dilated = vigra.filters.discDilation(closed, 2)
        plot_image_show(dilated, title="dilated seed")

        labels = vigra.analysis.labelImage(dilated)
        plot_image_show(labels, title="labels")

        pylab.figure()
        plot_image(data, title="seg vs real", alpha=0.5)
        plot_image(dilated, title="seg vs real", alpha=0.5)
        pylab.show()



    #igra.filters.discClosing()
    #http//hci.iwr.uni-heidelberg.de/vigra/doc/vigranumpy/index.html?highlight=dilate

    #dilation operator afterwards

    #vigra.analysis.labelVolume()
    #vigra.analysis.labelImage()

if __name__ == "__main__":
    extract()

if __name__ == "pyreport.main":
    extract()

