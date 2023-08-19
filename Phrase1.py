# Week 11
# Onur Tekiner
# Introduction to Python SUM/23
# 7/12/23
# Project (Phase 1)

import pandas as pd
import matplotlib.pyplot as plt

#downloading data from 'breast-cancer-wisconsin.data'
def read_to_pandas():
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    df = pd.read_csv('breast-cancer-wisconsin.data', names = col, na_values = '?')
    return df
#fill the null values in "A7" with mean of "A7"
def fill(df):
    #df[df['A7'].isnull()]=df['A7'].mean()
    df = df.fillna(round(df.mean(), 1))
    #create seperate table with only columns we need to analyze
    stat_data=df.loc[:,"A2":"A10"]
    return df,stat_data
#calculate mean,median,sd,var from A2 to A10
def stat(stat_data): 
    mean = round(stat_data.mean(), 1) 
    median = round(stat_data.median(), 1) 
    var = round(stat_data.var(), 1) 
    sd = round(stat_data.std(), 1)
    #print(stat_data)
    #print out each number with their attribute title
    for i in range(0,9):
        print("Attribute",i+2,"---------------------")
        print("\tMean:",(20-len("Mean"))*" ",mean[i])
        print("\tMedian:",(20-len("Median"))*" ",median[i])
        print("\tVariance:",(20-len("Variance"))*" ",var[i])
        print("\tStandard Deviation:",(20-len("Standart Deviation"))*" ",sd[i],"\n")


def grap(df):
    #slice to columns names that we need for histograms
    col = ["A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"]
    for i in col :
        #create histogram for each one of them
        plt.figure()
        plt.hist(df[i], edgecolor='black',bins=10, color = "blue", alpha = 0.5) 
        plt.title('Histogram of attribute {}'.format(i)) 
        plt.xlabel('Value of the attribute') 
        plt.ylabel('Number of data points') 
        plt.show()


def main():
    #apply every programs one by one
    df=read_to_pandas()
    df,stat_data=fill(df)
    stat(stat_data)
    grap(df)
    
main()



    
    
    
    
