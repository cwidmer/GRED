import numpy
import scipy
import scipy.linalg
import fit_ellipse_conic
import util
import cvxmod
import random
#from pylab import *
from util import conic_to_ellipse 
import pylab

import ransac

def ransac_ellipse(x,y):
    """ This defines the function ransac_ellipse
    """
    
#    ellipse = conic_to_ellipse(theta_)
#    return ellipse

class LeastSquaresEllipseModel():
    """
    linear system solved using linear least squares
    This class serves as an example that fulfills the model interface needed by the ransa function.
    """
    def __init__(self,input_columns,output_columns):
    #def __init__(self):
        self.input_columns = input_columns
        self.output_columns = output_columns
        #self.debug = debug
        print "setting up object"
        
    def fit(self, data):

        dat = phi_of_x(data)
        N = dat.shape[0]
        D = dat.shape[1]

        dat = cvxmod.matrix(dat)
        #### parameters

        # data
        X = cvxmod.param("X", N, D)

        #### varibales

        # parameter vector
        theta = cvxmod.optvar("theta", D)

        # simple objective 
        objective = cvxmod.atoms.norm2(X*theta)

        # create problem                                    
        p = cvxmod.problem(cvxmod.minimize(objective))
        p.constr.append(theta[0] + theta[1] == 1)
        
        ###### set values
        X.value = dat

        p.solve()
        
        cvxmod.printval(theta)

        theta_ = numpy.array(cvxmod.value(theta))
        #ellipse = conic_to_ellipse(theta_)

        #return ellipse
        return theta_
    
    
    def get_error(self, data, model):
        """
        get error for trained model wrt vector of points
        """
    
        err_per_point = numpy.zeros((data.shape[0]))
        for i, x in enumerate(phi_of_x(data)):
            err_per_point[i] = numpy.dot(x,model) ** 2
    
        
        return err_per_point


def phi_of_x(data):
    """
    get conic parameterization from x,y coordiantes
    """

    x, y = data[:,0], data[:,1]

    assert len(x) == len(y)

    N = len(x)
    D = 5

    dat = numpy.zeros((N, D))
    dat[:,0] = x*x
    dat[:,1] = y*y
    #dat[:,2] = x*y
    dat[:,2] = x
    dat[:,3] = y
    dat[:,4] = numpy.ones(N)

    print dat.shape

    return dat

def test():
#generate artificial data

        ellipse = util.Ellipse(0, 0, 0, 1, 1.5, 0)
    
        x_data, y_data = ellipse.sample_uniform(200);
       
        sort_idxs = numpy.argsort(x_data)
        x_sorted = x_data[sort_idxs]
    
    
        # add gaussian noise to original data
        for i in xrange(len(x_data)):
            
            x_data[i] += random.gauss(0, 0.2)
            y_data[i] += random.gauss(0, 0.2)
        
        num_errors = 40
        # create uniformly distributed errors
        err_x, err_y = numpy.zeros(num_errors), numpy.zeros(num_errors)

        for i in xrange(num_errors):
            err_x[i] = random.uniform(-3*ellipse.rx, 3*ellipse.rx)
            err_y[i] = random.uniform(-3*ellipse.rx, 3*ellipse.rx)
    
    
        # concat real and errorous data
        x_data = numpy.concatenate((x_data, err_x))
        y_data = numpy.concatenate((y_data, err_y))


        num_points = len(x_data)

        data = numpy.zeros((num_points, 2))
        data[:,0] = x_data
        data[:,1] = y_data
    
        model = LeastSquaresEllipseModel(input_columns=[0], output_columns=[1])
    
        theta = model.fit(data)
        model.get_error(data, theta)
        

        # run RANSAC algorithm
        ransac_fit, ransac_data = ransac.ransac(data, model, 20, 50, 7e-1, 20, debug=False,return_all=True)

        import ipdb
        ipdb.set_trace()
       
        ellipse_fit_epsilon = fit_ellipse_conic.fit_ellipse_eps_insensitive(x_data, y_data)
        
        #plot ellipse comparisons
        conic_to_ellipse(ransac_fit).plot_noshow(style='b-', label="ransac model fit")
        ellipse.plot_noshow(style='k-', label="ground truth")
        pylab.plot(x_data, y_data, 'k.', label="gaussian noise")
        pylab.plot(err_x, err_y, 'k.', label="uniform noise")
        
        ellipse_fit_epsilon.plot_noshow(style='g',label="epsilon insensitive fit")
        conic_to_ellipse(theta).plot_noshow(style='r-', label="leastsquare model fit")

        pylab.plot(x_data[ransac_data['inliers']], y_data[ransac_data['inliers']], 'bx', label='ransac data')
        
        pylab.legend(loc="best")
        pylab.xlim((-3,3))
        pylab.ylim((-3,3))
        
        pylab.show()
        pylab.legend()

        return ellipse_fit_ransac   

if __name__ == "__main__":
    test()