import numpy as np
from MathOp import MathOp
import matplotlib.pyplot as plt 
import math

pi = math.pi
iter_num = 100
learn_rate = 0.3
init_point = 6
plot_point = np.linspace(pi,2*pi,100,True)
plt.plot(plot_point,np.sin(plot_point))
plt.scatter(init_point, np.sin(init_point))
x = init_point
y = np.sin(x)
for i in range(iter_num):
    x = x - learn_rate * np.cos(x)
    y = np.sin(x)
    plt.scatter(x, y)
    plt.pause(0.5)
    pass
print(x)
print(y)
plt.show()

#plt.axis([0, 10, 0, 1])

#for i in range(10):
#    y = np.random.random()
#    plt.scatter(i, y)
#    plt.pause(0.05)

#plt.show()

