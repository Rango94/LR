from LR import LR

import numpy as np
import random as rd

def build_data(size):
    x=[]
    y=[]
    for i in range(size):
        x1=rd.random()
        x2=rd.random()
        x3=rd.random()
        x4=rd.random()
        x.append(np.array([x1,x2,x3,x4]))
        y.append([1 if x1*2+x2+x3*5+x4*(-2)>2 else 0])
    return np.array(x),np.array(y)

def lookup(y1,y2):
    for i,j in zip(y1,y2):
        print(i,j)

x_tr,y_tr=build_data(10000)
x_te,y_te=build_data(1000)

lr=LR(0.2)
lr.feed(x_tr,y_tr)

lookup(lr.predict(x_te),y_te)


