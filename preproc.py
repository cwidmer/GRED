
import vigra.filters
import data_processing
import numpy


def load_data():
    #readImage(filename, dtype = 'FLOAT', index = 0, order='') -> Image
    #TODO read tiff files
    vigra.impex.readImage()

def extract(data=None, *varargin):

    # This function localizes blob-like object using multi-scale hessian
    # aggregation. The algorithm has been described in 
    # [*} Xinghua Lou, X. Lou, U. Koethe, J. Wittbrodt, and F. A. Hamprecht. 
    # Learning to Segment Dense Cell Nuclei with Shape Prior. In The 25th 
    # IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2012), 2012.
    # 
    # Input:
    #       data:       image data
    # Additional options:
    #   scales = arg(varargin, 'scales', [0.9, 1.8]); -> smoothing scales
    #   thresholds = arg(varargin, 'thresholds', -0.5*[1, 1, 1]); -> hessian eigenvalue threshold
    # closing = arg(varargin, 'closing', true);
    # opening = arg(varargin, 'opening', true);
    # window = arg(varargin, 'window', 3);
    # conn = arg(varargin, 'conn', 0);
    # margin = arg(varargin, 'margin', 0);

    data = data_processing.get_data_example()

    width = 100
    height = 100
    depth = 100
    data = vigra.ScalarVolume((width, height, depth))
    #TODO: get data into right format
   
    scales = numpy.linspace(0.9, 0.3, 1.5)
    closing = True
    opening = False
    window = 3
    thresholds = numpy.array([-0.01, -0.1, -0.2])
    conn = 0
    margin = 0
    verbose = False
    ratios = numpy.array([1, 1, 0.25])
    sigmas = numpy.array([1, 2, 3])

    #seeds = arg(varargin, mstring('init'), true(size(data)))
    seeds = numpy.ones(data.size)

    for scale in scales:

        #TODO not sure what this is doing
        #if sum(seeds) == 0:
        #    println(mstring('early termination before sigma = %g'), scale)
        #    break

        # sigma
        if verbose:
            println(mstring('analyzing at sigma = %g'), scale)


        # smooth image at this scale
        tmp = vigra.filters.gaussianSmoothing(data, scale)

        # compute eigenvalues
        #eigenValues = vigra.filters.eigenValueOfHessianMatrix(tmp, sigma, 0.9 * numpy.array([1, 1, 1]), mask, seeds)
        tmp_sigma = numpy.eye(3)
        hessian = vigra.filters.hessianOfGaussianEigenvalues(data, tmp_sigma) #, out=None, sigma_d=0.0, step_size=1.0, window_size=0.0, roi=None)
        import ipdb
        ipdb.set_trace()

        # combine eigenvalue indicators: xor
        if ndims(data) == 3:
            seeds = _and(seeds, eigenValues(mslice[:], mslice[:], mslice[:], 1) < thresholds(1))
            seeds = _and(seeds, eigenValues(mslice[:], mslice[:], mslice[:], 2) < thresholds(2))
            seeds = _and(seeds, eigenValues(mslice[:], mslice[:], mslice[:], 3) < thresholds(3))
        elif ndims(data) == 2:
            seeds = _and(seeds, eigenValues(mslice[:], mslice[:], 1) < thresholds(1))
            seeds = _and(seeds, eigenValues(mslice[:], mslice[:], 2) < thresholds(2))
        end
    end

    #igra.filters.discClosing()
    #http//hci.iwr.uni-heidelberg.de/vigra/doc/vigranumpy/index.html?highlight=dilate

    #vigra.analysis.labelVolume()
    #vigra.analysis.labelImage()

if __name__ == "__main__":
    extract()

