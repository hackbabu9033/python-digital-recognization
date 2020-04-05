import csv
import os
import numpy as np
import xlwt 
from xlwt import Workbook 
from MathOp import MathOp
from NeuralNetWork import NN,MatMulSizeError

with open('train_100-0..csv') as f:
    train_data = csv.reader(f)
    train_data = list(train_data) 
    train_data.pop(0)

train = np.asarray(train_data)
size = np.shape(train)
x = train[:,1:size[1]]
sample_data = train[:,1:size[1]]
y = train[:,0]
x = x.astype(float)

#determine num of parameters for each layer
#+1 for bias terms
layer1_size = 3 + 1
#layer2_size = 30
pixel_count = size[1] 
label_count = 10

#divide data set into train set(70%) and test set(30%)
num_test = 12600
num_train = 29400


#random init parameter for each layer
theta_1 = MathOp.RandParams(layer1_size,pixel_count,0.12)
#theta_2 = MathOp.RandParams(layer2_size,layer1_size,0.12)
#theta_3 = MathOp.RandParams(label_count,layer2_size,0.12)
theta_2 = MathOp.RandParams(label_count,layer1_size,0.12)


#unroll parameter
nn_params =np.concatenate((theta_1.flatten(),theta_2.flatten()))

learn_rates = [0.05]
iter_num = 500
batch_size = 100
train_params = []
for rate in learn_rates:
    train_params,error = NN.gradient_descent(nn_params,rate,iter_num,num_train,batch_size,layer1_size,pixel_count,label_count)
    print('---cost function error---')
    print(error)
    pass
#get gradient from the minimize result 
bias = np.ones((size[0],1))
x = np.block([bias,x])
a = NN.forwardPropagation(x,train_params,layer1_size,pixel_count,label_count)
prediction = a[len(a)-1]
size = np.shape(prediction)
prediction_label = np.zeros((10,420))
y_label = np.zeros((10,420))
y = y.astype(int)
for i in range(size[1]):
    result = np.zeros((10))
    temp = prediction[:,i]   
    max_index = np.argmax(temp)
    prediction_label[max_index][i] = 1
    y_label[y[i]][i] = 1
    pass

error_flag = 0
#compute accurracy level
for i in range(size[1]):
    prediction_label[:,i] = np.absolute(prediction_label[:,i] - y_label[:,i])
    error_flag += np.sum(prediction_label[:,i] / 2)
    pass
print('---error rate for 420 data in train set---')
print(error_flag / 420)
    


#use test set to verify model accuracy
with open('train_test_set.csv') as f:
    test_data = csv.reader(f)
    test_data = list(test_data) 

test = np.asarray(test_data)
size = np.shape(test)
bias = np.ones((size[0],1))
x = test[:,1:size[1]]
test_data = test[:,1:size[1]]
x = np.block([bias,x])
y = test[:,0]
x = x.astype(float)
a = NN.forwardPropagation(x,train_params,layer1_size,pixel_count,label_count)
prediction = a[len(a)-1]
size = np.shape(prediction)
prediction_label = np.zeros((10,num_test))
error = np.zeros((10,num_test))
y_label = np.zeros((10,num_test))
y = y.astype(int)
for i in range(size[1]):
    result = np.zeros((10))
    temp = prediction[:,i]   
    max_index = np.argmax(temp)
    prediction_label[max_index][i] = 1
    y_label[y[i]][i] = 1
    pass

error_flag = 0
#compute accurracy level
for i in range(size[1]):
    error[:,i] = np.absolute(prediction_label[:,i] - y_label[:,i])
    error_flag += np.sum(error[:,i] / 2)
    pass
print('---error for test set data---')
print(error_flag / num_test)
    
#write predict result to excel file to compare result
# Workbook is created
wb = Workbook() 
  
# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1') 

for i in range(size[1]):
    predict_num = np.argmax(prediction_label[:,i])
    sheet1.write(i, 0, str(predict_num)) 
wb.save('predict_result.xls') 
    






