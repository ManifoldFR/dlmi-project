
## question : in which scale are we ??


import numpy as np

def IoU(seg1,seg2): # same as dice in the binary case
    intersection=np.sum(seg1*seg2)
    union=np.sum(seg1 + seg2 > 0)
    if union != 0 :
        return intersection/union
    else :
        return 0

def scaled_IoU(seg1,seg2):
    return IoU(seg1,seg2)/len(seg1)
    
def entropy(seg1,seg2): #negative sign is removed 
    histogram_seg=(seg1+seg2)/2
    return sum(histogram_seg==0.5)*0.5*np.log(0.5)

def scaled_entropy(seg1,seg2):
    return entropy(seg1,seg2)/len(seg1)




