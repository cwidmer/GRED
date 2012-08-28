
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
    thresholds = numpy.array([-0.5, -0.5, -0.5])
    conn = 0
    margin = 0
    verbose = False
    ratios = numpy.array([1, 1, 0.25])
    sigmas = numpy.array([1, 2, 3])

    #seeds = arg(varargin, mstring('init'), true(size(data)))
    seeds = data.size

    for scale in scales:

        #TODO not sure what this is doing
        #if sum(seeds) == 0:
        #    println(mstring('early termination before sigma = %g'), scale)
        #    break

        # sigma
        if verbose:
            println(mstring('analyzing at sigma = %g'), scale)


        # smooth image at this scale
        tmp = vigra.filters.gaussianSmoothing(data, 0.4)

        # compute eigenvalues
        #eigenValues = vigra.filters.eigenValueOfHessianMatrix(tmp, sigma, 0.9 * numpy.array([1, 1, 1]), mask, seeds)
        hessian = vigra.filters.hessianOfGaussianEigenvalues(data, scale) #, out=None, sigma_d=0.0, step_size=1.0, window_size=0.0, roi=None)
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

    # morphological operations
    if margin > 0:    # remove seeds at the boundaries/surfaces
        struc = false(size(seeds))
        if ndims(data) == 3:
            struc(mslice[margin + 1:end - margin], mslice[margin + 1:end - margin], mslice[margin + 1:end - margin]).lvalue = true
        elif ndims(data) == 2:
            struc(mslice[margin + 1:end - margin], mslice[margin + 1:end - margin]).lvalue = true
        end
        seeds = logical_and(seeds, struc)
    end

    if numel(closing) == 1:    # connect proximate seeds
        if closing:
            if ndims(data) == 3:
                seeds = imclose(seeds, true(1, 1, window))
                seeds = imclose(seeds, true(1, window, 1))
                seeds = imclose(seeds, true(window, 1, 1))
            elif ndims(data) == 2:
                seeds = imclose(seeds, true(1, window))
                seeds = imclose(seeds, true(window, 1))
                tmp = diag(true(window, 1))
                seeds = imclose(seeds, tmp)
                seeds = imclose(seeds, tmp(mslice[:], mslice[end:-1:1]))
            end
        end
    else:
        seeds = imclose(seeds, closing)
    end

    if numel(opening) == 1:    # remove noisy seeds
        if opening:
            if ndims(data) == 3:
                struc = false(window, window, window)
                c = (window + 1) / 2
                struc(mslice[:], c, c).lvalue = true
                struc(c, mslice[:], c).lvalue = true; print struc
                struc(c, c, mslice[:]).lvalue = true

                seeds = imopen(seeds, struc)
            elif ndims(data) == 2:
                struc = false(window, window)
                c = (window + 1) / 2
                struc(mslice[:], c).lvalue = true
                struc(c, mslice[:]).lvalue = true; print struc

                seeds = imopen(seeds, struc)
            end
        end
    else:
        seeds = imopen(seeds, opening)
    end

    if conn != 0:
        seeds = ctConnectedComponentAnalysis(uint16(seeds), false)
    end
    seeds = uint16(seeds)

if __name__ == "__main__":
    extract()
