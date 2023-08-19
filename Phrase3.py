# Week 12
# Onur Tekiner
# Introduction to Python SUM/23
# 7/27/23
# Project (Phase 3)

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

def finding_stats(final_df):
    error_24=final_df[(final_df["Predicted_Class"]==2)]
    error_24_vis=error_24[(error_24["Class"]==4)]
    error_24=error_24[(error_24["Class"]==4)].shape[0]
    #count_error_24=error_24[(error_24["Class"]==4)].shape[0]
    
    #for error_42       
    error_42=final_df[(final_df["Predicted_Class"]==4)]
    error_42_vis=error_42[(error_42["Class"]==2)]
    error_42=error_42[(error_42["Class"]==2)].shape[0]
    #count_error_42=error_42[(error_42["Class"]==2)].shape[0]
    
    #error_all
    error_all=error_42+error_24
    #pclass_2
    pclass_2=final_df[(final_df["Predicted_Class"]==2)].shape[0]
    #pclass_4
    pclass_4=final_df[(final_df["Predicted_Class"]==4)].shape[0]
    #class_all
    class_all=pclass_2 + pclass_4
    #error_B
    error_B=error_24/pclass_2*100
    #error_M
    error_M=error_42/pclass_4*100
    #error_T
    error_T=error_all/class_all*100
    
    return error_24,error_42,error_all,pclass_2,pclass_4,class_all,error_B,error_M,error_T,error_24_vis,error_42_vis
        
        
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
    
    #print("\nRandomly selected row",number1,"for centroid mu_2.") 
    #print("\nRandomly selected row",number2,"for centroid mu_4.")
    #print("\nProgram ended after",repeat,"iterations.")

    error_24,error_42,error_all,pclass_2,pclass_4,class_all,error_B,error_M,error_T,error_24_vis,error_42_vis=finding_stats(final_df)
    
    swapped=False
    #print("total indexes for two",final_df[final_df["Predicted_Class"]==2].index)
    #print("total indexes for four",final_df[final_df["Predicted_Class"]==4].index)
    
    print("Total errors:",round(error_T,1),"%")
    
    
    if error_T>50:
        swapped=True
        print("Clusters are swapped!")
        print("Swapping Predicted_Class")
        two_indexes=final_df[final_df["Predicted_Class"]==2].index
        four_indexes=final_df[final_df["Predicted_Class"]==4].index
        #final_df.iloc[two_indexes]["Predicted_Class"]=4
        final_df.iloc[four_indexes,2]=2
        final_df.iloc[two_indexes,2]=4
        error_24,error_42,error_all,pclass_2,pclass_4,class_all,error_B,error_M,error_T,error_24_vis,error_42_vis=finding_stats(final_df)
    if swapped==False:
        print("Clusters doesn't need swap!")
    #print(error_24)
    #print("predicted 2 but 4 totol:",error_24)
    #print(error_42)
    #print("predicted 4 but 2 totol:",error_42)
    #total error
    #print("how many error:",error_all)
    #all predicted 2 
    print("\nData points in Predicted Class 2:", pclass_2)
    #all predicted 4 
    print("Data points in Predicted Class 4:", pclass_4)
    
    print("\nError data points, Predicted Class 2:\n")
    print(error_24_vis)
    
    print("\nError data points, Predicted Class 4:\n")
    print(error_42_vis)
    
    
    #class_all
    print("\nNumber of all data points:", class_all)
    #total error
    print("\nNumber of error data points:",error_all)
    #error_b
    print("\nError rate for class 2:",round(error_B,1),"%")
    #error_M
    print("Error rate for class 4:",round(error_M,1),"%")
    #error_T
    print("Total error rate:", round(error_T,1),"%")
    
    #print("total indexes for two",final_df[final_df["Predicted_Class"]==2].index)
    #print("total indexes for four",final_df[final_df["Predicted_Class"]==4].index)
    #print("swapped",swapped)
    
    
main()
            
        
    
    
    
    
    
