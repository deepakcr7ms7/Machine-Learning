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
    for i in range(len(data_y)):
        if data_y[i]==0:
            data_y[i]=-1
    return data,data_x,data_y,test_x

def naive_baysian(data_x,data_y,test_x):

    # step 1 calculate prior probability P(Y)
    P_Y = []  # 1 by 2
    P_Y.append(len(data_y[data_y == -1]) / len(data_y))
    P_Y.append(len(data_y[data_y == 1]) / len(data_y))

    # split the data according to class
    data_x_w1 = data_x[data_y == -1]
    data_x_w2 = data_x[data_y == 1]

    # calculate mean and std of each class
    u_w1 = np.mean(data_x_w1, axis=0)  # 2 by1
    u_w2 = np.mean(data_x_w2, axis=0)

    u_w1 = np.reshape(u_w1, (2, 1))
    u_w2 = np.reshape(u_w2, (2, 1))

    std_w1 = np.std(data_x_w1, axis=0)  # 2 by 2
    std_w2 = np.std(data_x_w2, axis=0)

    def PDF(data,mean,std):
        #using PDF equation for Naive Bayes
        coe = 1 / (std * np.sqrt(2 * np.pi))
        sub = np.subtract(data, mean)
        exp = float(-1 / 2 * np.math.pow(sub/ std, 2))
        return float(coe * np.exp(exp))

    def Prob_x_given_w(data,mean,std):
        #as in naive we consider x1 and x2 are independant so we multiply thier probabilities to get the result
        class_1=PDF(data[0], mean[0], std[0])
        class_2 = PDF(data[1], mean[1], std[1])
        return class_1 * class_2

    def w1_posterior_func(x_value):
        p_x_given_w1=Prob_x_given_w(x_value,u_w1,std_w1) * P_Y[0]
        return p_x_given_w1
    def w2_posterior_func(x_value):
        p_x_given_w2=Prob_x_given_w(x_value,u_w2,std_w2) * P_Y[1]
        return p_x_given_w2

    y_pred = np.zeros((len(test_x)), dtype=int)

    for i in range(len(test_x)):
        x = test_x[i]
        x = np.reshape(x, (2, 1))
        if w1_posterior_func(x) > w2_posterior_func(x):
            y_pred[i] = -1
        else:
            y_pred[i] = 1

    return y_pred




if __name__=="__main__":
    data,data_x,data_y,test_x=data_extraction()
    y=naive_baysian(data_x,data_y,test_x)

    plt.scatter(data_x[data_y == -1][0:, 0], data_x[data_y == -1][0:, 1], c='red')
    plt.scatter(data_x[data_y == 1][0:, 0], data_x[data_y == 1][0:, 1], c='blue')
    plt.title("Training Data")
    plt.show()
    plt.scatter(test_x[y == -1][0:, 0], test_x[y == -1][0:, 1], c='red')
    plt.scatter(test_x[y == 1][0:, 0], test_x[y == 1][0:, 1], c='blue')
    plt.title("Result of Naive Baysian classifier")
    plt.show()


    y = np.reshape(y, (len(y), 1))
    DF = pd.DataFrame(y)
    DF.to_csv("NB.csv")
