import numpy as np
import math

class MathOp:
    @staticmethod
    def RandParams(row,column,epsilon_init):
        return np.random.rand(row,column)* 2 * epsilon_init - epsilon_init;      
    
    @staticmethod
    def sigmoid(z):
        '''x is a vector,theta is matrix, return a vector
ex: size(x)=[784,1],size(theta)=[layer1_size,784],then size(return)=[layer1_size,1]'''                 
        return 1 / (1 + np.exp(-z)) 
        





