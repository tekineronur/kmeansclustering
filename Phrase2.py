# Week 12
# Onur Tekiner
# Introduction to Python SUM/23
# 7/23/23
# Project (Phase 2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#downloading data from 'breast-cancer-wisconsin.data'
def read_to_pandas():
    col = ["Scn", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "Class"]
    df = pd.read_csv('Breast-Cancer-Wisconsin.csv', names = col, na_values = '?')
    return df
#fill the null values in "A7" with mean of "A7"
def fill(df):
    df[df['A7'].isnull()]=df['A7'].mean()
    #create seperate table with only columns we need to analyze
    stat_data=df.loc[:,"A2":"A10"]
    return stat_data,df

def initial_centroids():
    #apply every programs one by one
    df=read_to_pandas()
    stat_data,df=fill(df)
    #pulling random index number by numpy
    number1, number2=np.random.randint(0,df.shape[0], size=2)
    #returning random number via these index numbers
    mu_2=stat_data.iloc[number1]
    mu_4=stat_data.iloc[number2]
    return mu_2, mu_4, number1, number2

#Calculating distance of datas to random initial number
def distance(row,mu_2,mu_4):
    #applying Euclidian distance formula for each row
        dist_to_m2=np.sqrt(((row-mu_2)**2).sum()) 
        dist_to_m4=np.sqrt(((row-mu_4)**2).sum()) 
        #the distance of row is closer to mu_2 returns 2 otherwise 4
        if dist_to_m2 < dist_to_m4:  
            return 2
        else: 
            return 4
        
        
def main():
    #download data
    df=read_to_pandas()
    #fill null values and seperate value columns A2 to A10
    stat_data,df=fill(df)
    
    #Initial Step
    
    #receving our random rows and their index numbers
    mu_2,mu_4, number1, number2=initial_centroids()
    
    Predicted_Class=[]
    
    #Assign Step
    
    #applying our program which row closer to out initial point,
    #assign 2 or 4
    
    for row in range(stat_data.shape[0]):
        #applying our distance calculator program for
        # making list of assigns with 2 and 4's
        c=distance(stat_data.iloc[row],mu_2,mu_4)
        Predicted_Class.append(c)
    #adding this assignments in the dataframe
    df.insert(11,"Predicted_Class",Predicted_Class)
    #for calculate to iteration we assign repeat=0
    repeat=0
    
    #limited maximum 50 times iteration
    for iteration in range(50):
        
        repeat += 1
        
        #Recompute’ step, calculating new centroids with mean of numbers
        #of each cluster
        
        #returning index number of each row with predicted class 2 and 4
        mu_2_indexes=df[df["Predicted_Class"]==2].index 
        mu_4_indexes=df[df["Predicted_Class"]==4].index
        #taking mean of them for new centroids
        mu_2=stat_data.iloc[mu_2_indexes].mean() 
        mu_4=stat_data.iloc[mu_4_indexes].mean()  
        #assign next predicted class as Predicted_Class2
        Predicted_Class2=[] 
        #calculate distances again but this time with new centroid point
        for row in range(stat_data.shape[0]): 
            c=distance(stat_data.iloc[row],mu_2,mu_4) 
            #making another list of predicted class
            Predicted_Class2.append(c)
        #insert that one also in the dataframe
        df.insert(12,"Predicted_Class2",Predicted_Class2)
        #if there is no differences between
        #first predicted class and the next one
        #copy to last predicted class one to first one
        #and break the program
        if (df["Predicted_Class"]!= df["Predicted_Class2"]).sum() == 0 :
            df["Predicted_Class"]=df["Predicted_Class2"]
            break
        
        #if there is differences between first predicted class and next one
        #copy the last one as first one
        #and drop the last one in the dataframe so program doesn't crush
        else:
            df["Predicted_Class"]=df["Predicted_Class2"] 
            df=df.drop(["Predicted_Class2"], axis=1)
          
    final_df=df[["Scn","Class","Predicted_Class"]].astype(int)
    #information about initial number one for mu_2 
    print("\nRandomly selected row",number1,"for centroid mu_2.") 
    print("Initial centroid mu_2:") 
    print(stat_data.iloc[number1]) 
    #information about initial number two for mu_4 
    print("\nRandomly selected row",number2,"for centroid mu_4.") 
    print("Initial centroid mu_4:") 
    print(stat_data.iloc[number2]) 
    #about iteration 
    print("\nProgram ended after",repeat,"iterations.") 
    #final centroid of mu_2 
    print("\nFinal centroid mu_2:") 
    print(mu_2) 
    #final centroid of mu_4 
    print("\nFinal centroid mu_4:") 
    print(mu_4) 
    #print first 20 rows with Scn, Class, and Predicted Class columns 
    print("\nFinal cluster assignment:\n") 
    print(final_df.head(21))
    
    #print(final_df[final_df["Class"]==final_df["Predicted_Class"]].count())



    

    
main()
            
        
    
    
    
    
    
