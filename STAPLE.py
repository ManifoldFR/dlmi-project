# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:20:03 2020

@author: Philo
"""

import SimpleITK as sitk
import os
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, fixed
import staple
#import pandas as pd

#%run manifest.json


PROJECT_FOLDER="C:\\Users\\Philo\\Documents\\3A -- MVA\\DL for medical imaging\\retine\\dlmi-project\\"
os.chdir(PROJECT_FOLDER)
os.getcwd()


#list_DB=["stare","hrf","aria","drive","chasedb1"]
#DB_NAME="stare" 
DB_NAME="chasedb1" 
#DB_NAME="aria"
DATA_FOLDER = os.path.join(PROJECT_FOLDER,"data", DB_NAME,"images")
ANNOT_FOLDER=[os.path.join(PROJECT_FOLDER,"data", DB_NAME,"annotation 1"),
              os.path.join(PROJECT_FOLDER,"data", DB_NAME,"annotation 2")]
STAPLE_FOLDER=os.path.join(PROJECT_FOLDER,"data", DB_NAME,"STAPLE")


######## Data comparison - R output

list_annot1=os.listdir(ANNOT_FOLDER[0])
list_annot1=list_annot1[1:]

list_annot2=os.listdir(ANNOT_FOLDER[1])

im_nb=3

for im_nb in range(len(os.listdir(STAPLE_FOLDER))):
    st=plt.imread(os.path.join(STAPLE_FOLDER,os.listdir(STAPLE_FOLDER)[im_nb]))
    plt.imshow(st)
    
    a1=plt.imread(os.path.join(ANNOT_FOLDER[0],list_annot1[im_nb]))
    a2=plt.imread(os.path.join(ANNOT_FOLDER[1],list_annot2[im_nb]))
    #plt.imshow(a1)
    #plt.imshow(np.abs(a1-a2))
    #plt.imshow(np.abs(a1-st))
    #plt.imshow(np.abs(a2-st))
    
    union=(a1+a2>0)*1
    intersection=((a1==1) & (a2==1))*1
    #plt.imshow(union)
    plt.imshow(np.abs(union-st))
    print(os.listdir(ANNOT_FOLDER[0])[im_nb])
    print(os.listdir(ANNOT_FOLDER[1])[im_nb])
    print(os.listdir(STAPLE_FOLDER)[im_nb])
    print(np.unique(np.abs(union-st)))
    print(np.sum(np.abs(intersection-st)))
    print(np.sum(np.abs(union-st)))
    print(np.sum(np.abs(a1-st)))
    print(np.sum(np.abs(a2-st)))



##https://simpleitk.readthedocs.io/en/master/IO.html
##http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/03_Image_Details.html
##http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
#
#def plot_image(sitk_image):
#    plt.imshow(sitk.GetArrayViewFromImage(sitk_image))
#    plt.axis('off')
#    plt.show()
#
#
#def display_with_overlay(segmentation_number, slice_number, image, segs, window_min, window_max):
#    """
#    Display a CT slice with segmented contours overlaid onto it. The contours are the edges of 
#    the labeled regions.
#    """
#    img = image[:,:,slice_number]
#    msk = segs[segmentation_number][:,:,slice_number]
#    overlay_img = sitk.LabelMapContourOverlay(sitk.Cast(msk, sitk.sitkLabelUInt8), 
#                                              sitk.Cast(sitk.IntensityWindowing(img,
#                                                                                windowMinimum=window_min, 
#                                                                                windowMaximum=window_max), 
#                                                        sitk.sitkUInt8), 
#                                             opacity = 1, 
#                                             contourThickness=[2,2])
#    #We assume the original slice is isotropic, otherwise the display would be distorted 
#    plt.imshow(sitk.GetArrayViewFromImage(overlay_img))
#    plt.axis('off')
#    plt.show()
#
#    
#im_index=0
#
#image_np=plt.imread(os.path.join(DATA_FOLDER,os.listdir(DATA_FOLDER)[im_index]))
#image=sitk.ReadImage(os.path.join(DATA_FOLDER,os.listdir(DATA_FOLDER)[im_index]))
#plot_image(image)
#
#annotation_dict={1:[],2:[]}
#for i in range(1,3):
#    for filename in os.listdir(ANNOT_FOLDER[i-1]) : 
#        print(filename)
#        annot = sitk.ReadImage(os.path.join(ANNOT_FOLDER[i-1],filename))
#        annotation_dict[i].append(annot)
#        
#annotation_dict={1:[],2:[]}
#for i in range(1,3):
#    for filename in os.listdir(ANNOT_FOLDER[i-1]) : 
#        print(filename)       
#        reader = sitk.ImageFileReader()
#        reader.SetImageIO("PNGImageIO")
#        reader.SetFileName(os.path.join(ANNOT_FOLDER[i-1],filename))
#        annot = reader.Execute()
#        annotation_dict[i].append(annot)
#        
#plot_image(annotation_dict[1][0])
#annotation_np=[plt.imread(os.path.join(ANNOT_FOLDER[0],os.listdir(ANNOT_FOLDER[0])[im_index])),
#               plt.imread(os.path.join(ANNOT_FOLDER[1],os.listdir(ANNOT_FOLDER[1])[im_index]))]
#
#
#segmentations=[annotation_dict[1][im_index],annotation_dict[2][im_index]]
##segmentations=[annotation_dict[1],annotation_dict[2]]
#
## Use the STAPLE algorithm to obtain the reference segmentation. This implementation of the original algorithm
## combines a single label from multiple segmentations, the label is user specified. The result of the
## filter is the voxel's probability of belonging to the foreground. We then have to threshold the result to obtain
## a reference binary segmentation.
#foregroundValue = 1
#threshold = 0.95
#reference_segmentation_STAPLE_probabilities = sitk.STAPLE(segmentations, foregroundValue) 
#dir(reference_segmentation_STAPLE_probabilities)
#reference_segmentation_STAPLE_probabilities.GetSize()
#print(reference_segmentation_STAPLE_probabilities)
#
#list_t=[]
#list_nan_check=[]
#for thres in np.array(range(0,21))/20:
#    print(thres)
#    reference_segmentation_STAPLE_probabilities = sitk.STAPLE(segmentations, foregroundValue) 
#    t=np.array(reference_segmentation_STAPLE_probabilities)
#    list_t.append(t)
#    list_nan_check.append(sum(np.isnan(t))/len(t))
#    
#    # We use the overloaded operator to perform thresholding, another option is to use the BinaryThreshold function.
#    reference_segmentation_STAPLE = reference_segmentation_STAPLE_probabilities > thres
#    manual_plus_staple = list(segmentations)  
#    # Append the reference segmentation to the list of manual segmentations
#    manual_plus_staple.append(reference_segmentation_STAPLE)
#    plot_image(manual_plus_staple[2])
#
#
#
#interact(display_with_overlay, segmentation_number=(0,len(manual_plus_staple)-1), 
#         slice_number = (0, image.GetSize()[1]-1), image = fixed(image),
#         segs = fixed(manual_plus_staple), window_min = fixed(-1024), window_max=fixed(976));

         
         
         
         
         
         