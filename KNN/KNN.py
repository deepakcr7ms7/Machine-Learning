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


def KNN(data_x,data_y,test_x,k):
    #here k in number of neighbours we are considering
    y_prediction=[]
    y_confidence=[]
    for test in test_x:
        euclidian_dist=[]
        for train in data_x:
            #first we find the euclidian dist of test point from each training point
            e_dist=np.linalg.norm(train - test, ord=2)
            euclidian_dist.append(e_dist)
        #we need to find k nearest neighbours
        neighbours= np.argsort(euclidian_dist)[:k]

        #class of neighbours
        class_neighbours=data_y[neighbours][:k]

        #majority class

        class_bc=np.bincount(class_neighbours)
        majority_class=np.argmax(class_bc)

        #add this value to prediction
        y_prediction.append(majority_class)
        y_confidence.append(neighbours[majority_class]/len(neighbours))
    return y_prediction





if __name__=="__main__":
    data,data_x,data_y,test_x=data_extraction()
    y=KNN(data_x,data_y,test_x,3)
    y=np.array(y)

    # as np.bincount can't handle negative values now we will replace 0 with -1
    for i in range(len(data_y)):
        if data_y[i]==0:
            data_y[i]=-1
    for i in range(len(y)):
        if y[i]==0:
            y[i]=-1


# Scatter Plots
    plt.scatter(data_x[data_y == -1][0:, 0], data_x[data_y == -1][0:, 1], c='red')
    plt.scatter(data_x[data_y == 1][0:, 0], data_x[data_y == 1][0:, 1], c='blue')
    plt.title("Training Data")
    plt.show()

    plt.scatter(test_x[y == -1][0:, 0], test_x[y == -1][0:, 1], c='red')
    plt.scatter(test_x[y == 1][0:, 0], test_x[y == 1][0:, 1], c='blue')
    plt.title("Result of KNN classifier")
    plt.show()
#save result to CSV file
    y = np.reshape(y, (len(y), 1))
    DF = pd.DataFrame(y)
    DF.to_csv("KNN.csv")


