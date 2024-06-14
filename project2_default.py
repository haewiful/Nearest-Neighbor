import pandas as pd
import sys
import math
from scipy.stats import zscore
import scipy.stats
import time
import numpy as np

def main():
    # print initial info. and receive filename and 
    filename = input("Type in the name of the file to test: ")

    # read data & normalize it
    data = get_data(filename)

    class1_count=0
    for row in data:
        if row[0]==1:
            class1_count+=1
    
    if class1_count > len(data)/2:
        accuracy = class1_count / len(data) * 100
        print("class1")
    else:
        accuracy = (len(data)-class1_count) / len(data) * 100
        print("class2")
    print(accuracy)
    

def get_data(filename):
    data = pd.read_csv(filename, header=None, sep='  ', engine='python')

    # do z-normalization
    for col in data.columns:
        if col==0:
            continue
        data[col] = (data[col]-data[col].mean())/data[col].std(ddof=0)
    return data.to_numpy()

if __name__=="__main__":
    main()
