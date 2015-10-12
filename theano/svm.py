
import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T


class SVM(object):
    
    def __init__(self, input, n_in, n_out):
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        
        self.p_y_given_x = (T.dot(input, self.W) + self.b)

        self.y_pred = self.p_y_given_x
        
        self.params = [self.W, self.b]
        
        self.input = input

    def hinge(self,y):
        return T.maximum(0, 1-y)
    
    def cost(self,y):
        mar = y*self.p_y_given_x
        cost = self.hinge(mar).mean(axis=0).sum()
        return cost
    
    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        
        
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.sum(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

