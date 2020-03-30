import numpy as np
import math

class MathOp:
    @staticmethod
    def RandParams(row,column,epsilon_init):
        return np.random.rand(row,column) * 2 * epsilon_init - epsilon_init      
    
    @staticmethod
    def sigmoid(z):
        '''x is a vector,theta is matrix, return a vector
ex: size(x)=[pixel_count,1],size(theta)=[layer1_size,pixel_count],then size(return)=[layer1_size,1]'''                 
        return 1 / (1 + np.exp(-z)) 

    @staticmethod
    def sigmoid_diff(z):
        return z * (1 - z)

    @staticmethod
    def Relu(z):
        if z > 0:
            return z
        else:
            0

    @staticmethod
    def Relu_diff(z):
        if z > 0:
            return 1
        else:
            0
            
    @staticmethod
    def add(x,y):
        return x + y

    @staticmethod 
    def minus(x,y):
        return x - y
        





