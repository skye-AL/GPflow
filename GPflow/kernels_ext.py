import tensorflow as tf
import numpy as np
from .param import Param, Parameterized
from . import transforms
from kernels import Kern


class StdPeriodic(Kern):
    """
    Standard periodic kernel
    """
    def __init__(self, input_dim, variance=1.0, periods=None, lengthscales=None, active_dim = None, ARD1=False,
                 ARD2=False):
        Kern.__init__(self, input_dim, active_dim)
        self.variance = Param(variance, transforms.positive)
        if ARD1:
            if periods is None:
                periods = np.ones(input_dim)
            else:
                period = np.asarray(periods)
                assert period.size == input_dim, "bad number of periods"
            self.periods = Param(period, transforms.positive)
            self.ARD1 = True
        else:
            if periods is None:
                period = 1.0
            else:
                period = np.asarray(periods)
                assert period.size == 1, "Only one period needed for non-ARD kernel"
            self.periods = Param(period, transforms.positive)
            self.ARD1 = False

        if ARD2:
            if lengthscales is None:
                lengthscales = np.ones(input_dim)
            else:
                lengthscales = np.asarray(lengthscales)
                assert lengthscales.size == input_dim, "bad number of lengthscales"
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD2 = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            else:
                lengthscales = np.asarray(lengthscales)
                assert lengthscales.size == 1, "Only one lengthscale needed for non-ARD kernel"
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD2 = False

    def Kdiag(self, X):
        zeros = X[:,0]*0
        return zeros + self.variance

    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X
        base = np.pi * (tf.expand_dims(X,1)-tf.expand_dims(X2,0)) / self.periods
        exp_dist = tf.exp(
            -0.5 * tf.reduce_sum(tf.square(tf.sin(base) / self.lengthscales), reduction_indices=2))
        return self.variance * exp_dist


class Stationary3DAngle(Kern):
    """
    Base class for kernels that are based on the 3D angle between two vectors

        alpha = acos((xx')/(||x||||x'||))

    """

    def __init__(self, variance=1.0, lengthscales=None, active_dims=None):
        """
        variance is the (initial) value for the variance parameter
        lengthscale is the initial value for the lengthscale parameter
         --defaults to 1.0.
        active_dims is a list of length 3 which controls thwich columns of X are used.
        """
        Kern.__init__(self, 3, active_dims)
        self.variance = Param(variance, transforms.positive)
        if lengthscales is None:
            lengthscales = 1.0
        self.lengthscales = Param(lengthscales, transforms.positive)
        self.ARD = False

    def angular_dist(self, X, X2=None):
        """

        :param X: matrix of size mx3
        :param X2: matrix of size lx3
        :return: matrix of size mxl containing the angle between the vectors
        """
        if X2 is None:
            XXT = tf.matmul(X, tf.transpose(X))  # matrix of size mxm
            xnorm = tf.sqrt(tf.reduce_sum(tf.square(X), 1))  # matrix of size mx1
            Xnorm, XnormT = tf.meshgrid(xnorm, xnorm, indexing='ij')  # matrix of shape mxm
            CosAlpha = tf.div(XXT,tf.mul(Xnorm,XnormT))
            Alpha = tf.acos(CosAlpha)/self.lengthscales  # can result in nan due to numerical errors
            Zeros = Alpha[:, 0] * 0.0
            return tf.batch_matrix_set_diag(Alpha, Zeros)
        else:
            XX2T = tf.matmul(X, tf.transpose(X2)) #matrix of size mxl
            xnorm = tf.sqrt(tf.reduce_sum(tf.square(X), 1)) #matrix of size mx1
            x2norm = tf.sqrt(tf.reduce_sum(tf.square(X2), 1)) #matrix of size lx1

            #do meshgrid
            Xnorm, X2norm = tf.meshgrid(xnorm, x2norm, indexing='ij') #matrix of shape mxl
            CosAlpha = XX2T/(Xnorm*X2norm)
            CosAlpha = tf.maximum(tf.minimum(CosAlpha, 1),-1) # not good for gradient
            #CosAlpha = tf.sub(CosAlpha, 1e-10) #make sure entries are in range [-1, 1]
            return tf.acos(CosAlpha)/self.lengthscales

    def Kdiag(self, X):
        zeros = X[:, 0] * 0
        return zeros + self.variance

class Exponential3DAngle(Stationary3DAngle):
    """
    The exponential kernel for 3D angles
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-self.angular_dist(X, X2))


class Stationary3DScalarProduct(Kern):
    """
    Base class for kernels that are based on the scalar product between two vectors

        scalar = (xx')

    """

    def __init__(self, variance=1.0, lengthscales=None, active_dims=None):
        """
        variance is the (initial) value for the variance parameter
        lengthscale is the initial value for the lengthscale parameter
         --defaults to 1.0.
        active_dims is a list of length 3 which controls thwich columns of X are used.
        """
        Kern.__init__(self, 3, active_dims)
        self.variance = Param(variance, transforms.positive)
        if lengthscales is None:
            lengthscales = 1.0
        self.lengthscales = Param(lengthscales, transforms.positive)
        self.ARD = False

    def scalar_product(self, X, X2=None):
        """

        :param X: matrix of size mx3
        :param X2: matrix of size lx3
        :return: matrix of size mxl containing the scalar product between the vectors
        """
        if X2 is None:
            XXT = tf.matmul(X, tf.transpose(X))  # matrix of size mxm
            xnorm = tf.sqrt(tf.reduce_sum(tf.square(X), 1))  # matrix of size mx1
            Xnorm, XnormT = tf.meshgrid(xnorm, xnorm, indexing='ij')  # matrix of shape mxm
            XXTnorm = tf.div(XXT, tf.mul(Xnorm, XnormT))
            #Ones = XXTnorm[:, 0] * 1.0
            #return tf.batch_matrix_set_diag(XXTnorm, Ones)
            return XXTnorm
        else:
            XX2T = tf.matmul(X, tf.transpose(X2))  # matrix of size mxl
            xnorm = tf.sqrt(tf.reduce_sum(tf.square(X), 1))  # matrix of size mx1
            x2norm = tf.sqrt(tf.reduce_sum(tf.square(X2), 1))  # matrix of size lx1

            # do meshgrid
            Xnorm, X2norm = tf.meshgrid(xnorm, x2norm, indexing='ij')  # matrix of shape mxl
            XX2Tnorm = XX2T / (Xnorm * X2norm)
            return XX2Tnorm

    def Kdiag(self, X):
        zeros = X[:, 0] * 0
        return zeros + self.variance

class Exponential3DScalarProduct(Stationary3DScalarProduct):
    """
    The exponential kernel for 3D angles
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-(1.0-self.scalar_product(X, X2))/self.lengthscales)



