import numpy as np
import math
class LR:

    def __init__(self,lr):
        self.flag=True
        self.lr=lr
    def feed(self,x,y):
        batch_size,feature_num=x.shape
        if self.flag:
            self.flag=False
            self.Theta=np.zeros(feature_num)
        refresh_theta=np.zeros(feature_num)

        for idx,each_f in enumerate(x):
            logic=self.sigmoid(np.dot(self.Theta,each_f))
            refresh_theta+=self.lr*(y[idx]-logic)*each_f
            if idx%10==0:
                self.Theta+=refresh_theta/10
                refresh_theta=np.zeros(feature_num)
                print(self.Theta)
        print(self.Theta)
    
    def predict(self,x):
        if self.flag:
            print('feed before predict')
        else:
            y=[]
            for idx,each_f in enumerate(x):
                y.append(int(self.sigmoid(np.dot(self.Theta,each_f))+0.5))
            return np.array(y)

    def sigmoid(self,x):
        return 1/(1+math.pow(math.e,-x))
