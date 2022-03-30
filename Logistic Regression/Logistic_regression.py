import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def data_extraction():

    df=pd.read_csv("training_set.csv")
    data=np.array(df)
    test=pd.read_csv("test_set.csv")
    test_x=np.array(test,dtype=float)
    data_x=np.array(data[:,[0,1]],dtype=float) #1000 by 2
    data_y=np.array(data[:,2],dtype=int) #1 by 1000
    """
    for i in range(len(data_y)):
        if data_y[i]==0:
            data_y[i]=-1
    """
    return data,data_x,data_y,test_x
def sigmoid(x):
    return 1/(1+np.exp(-x))
def logistic_regression_training(data_x,data_y,learning_rate,iterations):
    data_y=data_y.T
    m,n=data_x.shape   #n,m

    W=np.zeros((n,1))
    B=0
    cost_func=[]

    for i in range (iterations):
        #probabilistic predictions
        Prediction=sigmoid(np.dot(W.T,data_x.T)+B)  #A

        #Cost Function

        cost=-(1/m)*(np.sum(data_y*np.log(Prediction)+(1-data_y)*np.log(1-Prediction)))

        # for gradient descent we need dW and dB
        dW=(1/m)*np.dot(Prediction-data_y,data_x)
        dB=(1/m)*np.sum(Prediction-data_y)

        #new value of W and B

        W=W- learning_rate*dW.T

        B=B- learning_rate*dB
        cost_func.append(cost)
    return W, B, cost_func


def prediction(W,B,test_x):
    Z=np.dot(W.T,test_x.T)+B
    y_prediction=sigmoid(Z)
    y_prediction=y_prediction>0.5

    y_prediction=np.array(y_prediction,dtype='int64')

    return y_prediction.T


if __name__=="__main__":
    data,data_x,data_y,test_x=data_extraction()
    W, B, cost_func=logistic_regression_training(data_x,data_y,0.005,1000)

    #we will see if the gradient descent  minimized the cost function or not
    plt.plot(cost_func)
    plt.title("Effect of gradient descent on cost function")
    plt.show()

    y=prediction(W,B,test_x)
    y=np.reshape(y,(len(test_x)))

    plt.scatter(data_x[data_y == 0][0:, 0], data_x[data_y == 0][0:, 1], c='red')
    plt.scatter(data_x[data_y == 1][0:, 0], data_x[data_y == 1][0:, 1], c='blue')
    plt.title("Training Data")
    plt.show()

    plt.scatter(test_x[y == 0][0:, 0], test_x[y == 0][0:, 1], c='red')
    plt.scatter(test_x[y == 1][0:, 0], test_x[y == 1][0:, 1], c='blue')
    plt.title("Test Data")
    plt.show()

    for i in range(len(y)):
        if y[i]==0:
            y[i]=-1

    y=np.reshape(y,(len(y),1))
    DF=pd.DataFrame(y)
    DF.to_csv("LR.csv")

