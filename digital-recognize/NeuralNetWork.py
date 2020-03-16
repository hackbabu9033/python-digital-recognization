import numpy as np
import csv
from MathOp import MathOp
import matplotlib.pyplot as plt 
class MatMulSizeError(Exception):
    pass

class NN():
    @staticmethod
    def costFunction(nn_params,train,data_size,layer1_size,layer2_size):
        #get x and y
        data = train.reshape(data_size,785)
        size = np.shape(data)
        x = data[:,1:size[1]]
        y = data[:,0]
        
        size = np.shape(x)
        m = size[0]
        y = y.astype(int)
        y_label = np.zeros((layer2_size,size[0]))
        y_label_size = np.shape(y_label)
        for i in range(y_label_size[1]):
            y_label[y[i]][i] = 1
            pass
        a = NN.forwardPropagation(x,nn_params,layer1_size,layer2_size)
        h = a[len(a)-1]

        #use outputlayer to compute J
        j = 0
        for i in range(m):
            sample_cost = np.sum((y_label[:,i].T)*(np.log(h[:,i]))+((1-y_label[:,i]).T)*np.log(1-(h[:,i])))
            j = j + sample_cost
            pass
        j = j*(-1/m);
        return j

    def forwardPropagation(x,nn_params,layer1_size,layer2_size):
        '''x is input layer 
        nn_params ->每一層的參數向量化的結果
'''
        nn_params = np.asarray(nn_params)
        nn_params = nn_params.flatten()
        theta = []
        theta1 = nn_params[0:(layer1_size*784)].reshape(layer1_size,784)
        theta2 = nn_params[(layer1_size*784):].reshape(layer2_size,layer1_size)
        theta.append(theta1)
        theta.append(theta2)
        a = np.transpose(x)
        hidden_size = len(theta)
        # 紀錄每一個隱藏層的output
        outputs = []
        for param in theta:           
            param_size = np.shape(param)
            size_a = np.shape(a)
            a = np.asarray(a)
            a = a.astype(float)
            if(param_size[1]!=size_a[0]):
                raise MatMulSizeError
            z = np.dot(param,a)
            a = MathOp.sigmoid(z)
            outputs.append(a)
        return outputs

    def backwardPropagation(nn_params,train,batch_size,layer1_size,layer2_size):        
        #reshape nn_params to theta1 and theta2
        theta = []
        theta1 = nn_params[0:(layer1_size*784)].reshape(layer1_size,784)
        theta2 = nn_params[(layer1_size*784):].reshape(layer2_size,layer1_size)
        theta.append(theta1)
        theta.append(theta2)

        #get x and y
        data = train.reshape(batch_size,785)
        size = np.shape(data)
        x = data[:,1:size[1]]
        y = data[:,0]

        size = np.shape(x)
        m = size[0]
        y = y.astype(int)
        y_label = np.zeros((layer2_size,size[0]))
        y_label_size = np.shape(y_label)
        for i in range(y_label_size[1]):
            y_label[y[i]][i] = 1
            pass
        a = NN.forwardPropagation(x,nn_params,layer1_size,layer2_size)
        h = a[len(a)-1]       

        #update gradiant 
        #計算第3層跟第2層的delta以及梯度(第一層是輸入層)
        theta2_size = np.shape(theta[1])
        theta1_size = np.shape(theta[0])

        theta_2_grad = np.zeros(theta2_size)
        delta_2 = np.zeros((theta2_size))
        error_3 = h-y_label
        for i in range(m):
           error = np.reshape(error_3[:,i],(layer2_size,1))
           active = np.reshape(a[0][:,i],(1,layer1_size))
           delta_2 = delta_2 + (error @ active)
           pass
        theta_2_grad = delta_2/m

        #計算第二層
        error_2 = (theta[1].T@error_3)*a[0]*(1-a[0])
        theta_1_grad = np.shape(theta1_size)
        delta_1 = np.zeros((theta1_size))
        for i in range(m):
           error = np.reshape(error_2[:,i],(layer1_size,1))
           active = np.reshape(x[i,:],(1,784))
           active = active.astype(float)
           delta_1 = delta_1 + (error @ active)
        theta_1_grad = delta_1/m       

        costfun_grad = np.concatenate((theta_1_grad.flatten(),theta_2_grad.flatten()))
        return costfun_grad

    def gradient_check(nn_params, epsilson, train):
        size = np.shape(nn_params)
        nn_params_plus = np.zeros((size))
        nn_params_minus = np.zeros((size))
        number_gradient = np.zeros((size))
        for i in range(size[0]):
            nn_params_plus = np.copy(nn_params)
            nn_params_minus = np.copy(nn_params)
            nn_params_plus[i] = nn_params_plus[i] + epsilson
            nn_params_minus[i] = nn_params_minus[i] - epsilson
            number_gradient[i] = NN.costFunction(nn_params_plus,train)-NN.costFunction(nn_params_minus,train)
            number_gradient[i] = number_gradient[i]/(2*epsilson)
            pass
        return number_gradient

    def gradient_descent(nn_params,learn_rate,iter_num,
                         train_size,batch_size,layer1_size,layer2_size):
        #open file
        with open('train_train_set.csv') as f:
            train_data = csv.reader(f)
            train_data = list(train_data)
            #train_data.pop(0)
            train_data = np.asarray(train_data)

        iterpoint = []   
        error_values = []
        #batch gradient
        for j in range(iter_num):
            batch_mask = np.random.choice(train_size,batch_size)
            batch_train_set = train_data[(batch_mask),:]
            batch_train_set = np.asarray(batch_train_set)
            batch_train_set = batch_train_set.flatten()
            batch_train_set = batch_train_set.astype(float)
            gradients = NN.backwardPropagation(nn_params,batch_train_set,batch_size,layer1_size,layer2_size)
            nn_params = nn_params - learn_rate * gradients
            #compute error for each 100 iterate and draw
            if(j % 100 == 0):
                iterpoint.append(j)
                error = NN.costFunction(nn_params,batch_train_set,batch_size,layer1_size,layer2_size)
                error_values.append(error)
            pass    
        error = NN.costFunction(nn_params,batch_train_set,batch_size,layer1_size,layer2_size)
        plt.plot(iterpoint,error_values)
        plt.show()
        return nn_params,error

