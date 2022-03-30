import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
"""
Bayes Classifier
"""
def data_extraction():

    df=pd.read_csv("training_set.csv")
    data=np.array(df)
    test=pd.read_csv("test_set.csv")
    test_x=np.array(test,dtype=float)
    data_x=np.array(data[:,[0,1]],dtype=float) #1000 by 2
    data_y=np.array(data[:,2],dtype=int) #1 by 1000
    for i in range(len(data_y)):
        if data_y[i]==0:
            data_y[i]=-1
    return data,data_x,data_y,test_x

def baysian(data_x,data_y,test_x):

    #split the data according to class
    data_x_w1=data_x[data_y==-1]
    data_x_w2=data_x[data_y==1]
    """
    print(data_x_w1)
    print(data_x_w1.shape)
    """
    # step 1 calculate prior probability P(W)
    P_W = [] #1 by 2
    P_W.append(len(data_y[data_y == -1]) / len(data_y))
    P_W.append(len(data_y[data_y == 1]) / len(data_y))

    #step 2 get PDF of

    #calculate mean and covariance  of each class

    u_w1=np.mean(data_x_w1,axis=0) # 2 by1
    u_w2=np.mean(data_x_w2,axis=0)

    u_w1=np.reshape(u_w1,(2,1))
    u_w2 = np.reshape(u_w2, (2, 1))

    cov_w1=np.cov(data_x_w1,rowvar=False) #2 by 2
    cov_w2 =np.cov(data_x_w2,rowvar=False)
    """
    print("w1 covar", cov_w1)
    print("w2 covar", cov_w2)
    """


    # we need det and inverse of covariance matrix
    cov_w1_inv = np.linalg.inv(cov_w1)
    cov_w2_inv = np.linalg.inv(cov_w2)


    cov_w1_det = np.linalg.det(cov_w1)
    cov_w2_det = np.linalg.det(cov_w2)

    #print("deteminants",cov_w1_det,cov_w2_det)



    def PDF(data, mean,cov_inv, cov_det):
        #using Gaussian PDF formula given in textbook
        c = 1 / (2 * np.pi * cov_det)

        sub=np.subtract(data,mean)
        sub_T=sub.transpose()

        exp = float(-1 / 2 * np.matmul(np.matmul(sub_T, cov_inv),sub))
        return float(c * np.exp(exp))

    def w1_posterior_func(x_value):
        p_x_given_w1= PDF(x_value, u_w1, cov_w1_inv, cov_w1_det) * P_W[0]

        return p_x_given_w1
    def w2_posterior_func(x_value):
        p_x_given_w2= PDF(x_value, u_w2, cov_w2_inv, cov_w2_det) * P_W[1]
        return p_x_given_w2



    y_pred=np.zeros((len(test_x)),dtype=int)

    for i in range(len(test_x)):
        x=test_x[i]
        x=np.reshape(x,(2,1))
        if w1_posterior_func(x) > w2_posterior_func(x):
            y_pred[i] = -1
        else:
            y_pred[i] = 1
    return y_pred




if __name__=="__main__":
    data,data_x,data_y,test_x=data_extraction()
    print(data_y.shape)
    y=baysian(data_x,data_y,test_x)

    plt.scatter(data_x[data_y == -1][0:, 0], data_x[data_y == -1][0:, 1], c='red')
    plt.scatter(data_x[data_y == 1][0:, 0], data_x[data_y == 1][0:, 1], c='blue')
    plt.title("Training Data")
    plt.show()

    plt.scatter(test_x[y == -1][0:, 0], test_x[y == -1][0:, 1], c='red')
    plt.scatter(test_x[y == 1][0:, 0], test_x[y == 1][0:, 1], c='blue')
    plt.title("Result of Baysian classifier")

    plt.show()

    #save result to CSV file
    y = np.reshape(y, (len(y), 1))
    DF = pd.DataFrame(y)
    DF.to_csv("BR.csv")
