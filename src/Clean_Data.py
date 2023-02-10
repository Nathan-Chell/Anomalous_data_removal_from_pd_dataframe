#Detect and remove anomalies in the data


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

def Load_dataset():
    # Load the dataset
    np.set_printoptions(suppress=True)
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    #This is required to format the data, the website stores it over multiple lines
    #So this is used to combine them onto one row.
    numpy_data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :3]])

    # Create a dataframe
    column_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
    df_data = pd.DataFrame(numpy_data, columns=column_names)
    
    return df_data

def Scatter_plot(df_data):
    
    # Scatter plot
    #fig variable isnt used but is required for the plot to show
    fig, ax = plt.subplots(figsize = (18,10))
    ax.scatter(df_data['INDUS'], df_data['TAX'])
    
    # x-axis label
    ax.set_xlabel('(Proportion non-retail business acres)/(town)')
    
    # y-axis label
    ax.set_ylabel('(Full-value property-tax rate)/( $10,000)')
    plt.show()

def Z_Score(df_data):
    
    # Z-Score can be used to detect outliers
    # All data that falls between 3 standard deviations isn't considered an outlier
    Z_Score = np.abs(stats.zscore(df_data['DIS']))
        
    outliers = np.where(Z_Score > 3)
        
    return outliers[0]

def Interquartile_Range(df_data):
    
    #View data between the 25th and 75th percentile
    #This data represents a close distribution of the data
    
    Q1 = df_data.quantile(0.25, interpolation='midpoint')
    Q3 = df_data.quantile(0.75, interpolation='midpoint')
    
    IQR = Q3 - Q1
    
    #For a wider view of the data, we can view data between 2.7 standard deviations
    
    upper_limit = (Q3 + 1.5 * IQR)
    lower_limit = (Q1 - 1.5 * IQR)
    
    upper_points = np.where(df_data['DIS'] >= upper_limit['DIS'])
    lower_points = np.where(df_data['DIS'] <= lower_limit['DIS'])
    
    outliers = np.concatenate((upper_points, lower_points), axis=None)
    
    return outliers

def Remove_outliers(df_data, outliers):
    
    #Remove the outliers from the dataset
    df_data = df_data.drop(outliers)
    
    return df_data

def main():
    
    df_data = Load_dataset()
    
    #Plot data to view outliers
    #Scatter_plot(df_data)
    
    #Compute outliers
    outliers = Z_Score(df_data)
    #outliers = Interquartile_Range(df_data)
    
    print("Dataframe size before outlier removal: {}".format(df_data.shape))
    
    df_data = Remove_outliers(df_data, outliers)
    
    print("Dataframe size after outlier removal: {}".format(df_data.shape))
    
    

if __name__ == "__main__":
    main()