
# Comparing different model types
# for Binary classification on medical data
# sklearn's breast cancer dataset will be used

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


sns.set_style('dark')
mpl.style.use(['https://gist.githubusercontent.com/BrendanMartin/01e71bb9550774e2ccff3af7574c0020/raw/6fa9681c7d0232d34c9271de9be150e584e606fe/lds_default.mplstyle'])
mpl.rcParams.update({"figure.figsize": (8,6), "axes.titlepad": 22.0})
    
    
def Get_Data():
    return load_breast_cancer()
    
def Show_Data(dataset):
    
    # Printing the dataset description
    (unique, counts) = np.unique(dataset['target'], return_counts=True)
    
    print('Target variables: ', dataset['target_names'])
    print('Unique values of the target variable:', unique)
    print('Counts of the unique values of the target variable:', counts)
    
    return unique, counts

def Plot_data(dataset, Total_samples):
    
    # Plotting the target variable counts
    sns.barplot(x=dataset['target_names'], y=Total_samples)
    plt.title('Target variable counts in dataset')
    plt.show()

def Split_Data(X, Y):
    
    # Splitting the dataset into the Training set and Test set
    # 75% of the data will be used for training and 25% for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    return X_train, X_test, Y_train, Y_test

def Logistic_Regression(X, Y):
    
    standardizer = StandardScaler()
    X = standardizer.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = Split_Data(X, Y)
    
    model = LogisticRegression()
    #Train the model
    model.fit(X_train, Y_train)
    
    predictions = model.predict(X_test)

    TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()

    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)

    accuracy =  (TP+TN) /(TP+FP+TN+FN)

    print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))

    
    
def main():

    dataset = Get_Data()
    Sample_options, Total_samples = Show_Data(dataset)
    #Plot_data(dataset, Total_samples)
    
    X, Y = dataset['data'], dataset['target']
    
    #Run just Logistic regression
    #Logistic_Regression(X, Y)
    
    #Compare different models
    #Add models to the list
    models = {}
    
    models['Logistic Regression'] = LogisticRegression()
    models['Support Vector Machine'] = LinearSVC()
    models['Decision Tree'] = DecisionTreeClassifier()
    models['Random Forest'] = RandomForestClassifier()
    models['Naive Bayes'] = GaussianNB()
    models['K-Nearest Neighbors'] = KNeighborsClassifier()
    
    accuracy, precision, recall = {}, {}, {}
    X_train, X_test, Y_train, Y_test = Split_Data(X, Y)
    
    #loop through all the models, training and evaluating them all
    for key in models.keys():
        models[key].fit(X_train, Y_train)
        predictions = models[key].predict(X_test)
        
        accuracy[key] = accuracy_score(predictions, Y_test)
        precision[key] = precision_score(predictions, Y_test)
        recall[key] = recall_score(predictions, Y_test)
        
    
    df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
    df_model['Accuracy'] = accuracy.values()
    df_model['Precision'] = precision.values()
    df_model['Recall'] = recall.values()

    print(df_model)
    
    ax  = df_model.plot.bar(rot=45)
    ax.legend(ncol= len(models.keys()), bbox_to_anchor=(0, 1), loc='lower left', prop={'size': 14})
    plt.tight_layout()
    plt.show()

    

    
if __name__ == "__main__":
    main()