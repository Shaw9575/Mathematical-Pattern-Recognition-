# count function 
import numpy as np

def CNT(data, target):
    m = data.shape[0]
    count = 0
    for i in range(0,m):
        if data[i] == target:
            count = count+1
    return(count)
