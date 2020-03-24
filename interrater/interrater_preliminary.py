

import os 
from PIL import Image
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from utils import interrater_metrics
import pickle as pkl
from keras.preprocessing.image import load_img,img_to_array, array_to_img


#PROJECT_FOLDER = os.getcwd()
PROJECT_FOLDER="C:\\Users\\Philo\\Documents\\3A -- MVA\\DL for medical imaging\\retine\\dlmi-project\\"
INTERRATER_FOLDER=os.path.join(PROJECT_FOLDER, "data", 'interrater_data')
DB_list = ["aria","chasedb1","stare"]
nb_annot = 2

## Build dictionnary of annotations 
#seg_dict = {}
#for DB in DB_list : 
#    DATA_FOLDER = os.path.join(PROJECT_FOLDER,"data", DB)
#    seg_dict[DB]={}
#    for i in range(nb_annot) :
#        seg_dict[DB][i]={"img":[],"files":[]}
#        folder = os.path.join(DATA_FOLDER,str("annotation "+str(i+1)))
#        for file in os.listdir(folder):
#            if (".tif" in file) or (".png" in file) :
##                seg_dict[DB][i]["img"].append(Image.open(os.path.join(folder,file)))
#                img = plt.imread(os.path.join(folder,file))
#                if np.max(img) > 1 :
#                    img = img / 255
#                seg_dict[DB][i]["img"].append(img)
#                seg_dict[DB][i]["files"].append(file[:-4])
#pkl.dump(seg_dict, open(os.path.join(INTERRATER_FOLDER,'seg_dict_double_annotation.pkl'), 'wb'))

seg_dict = pkl.load(open(os.path.join(INTERRATER_FOLDER,'seg_dict_double_annotation.pkl'), 'rb'))

## Create a dictionnary of images and file names
#img_dict={}
#for DB in DB_list : 
#    DATA_FOLDER = os.path.join(PROJECT_FOLDER,"data", DB)
#    img_dict[DB]={}
#    img_dict[DB]={"img":[],"files":[]}
#    folder = os.path.join(DATA_FOLDER,"images")
#    if len(os.listdir(folder))==len(seg_dict[DB][1]["files"]): 
#        for file in os.listdir(folder):
#            if (".tif" in file) or (".png" in file) or ("jpg" in file):
#                img_dict[DB]["img"].append(Image.open(os.path.join(folder,file)))
#                img_dict[DB]["files"].append(file[:-4])
#    else : #stare dabase
#        list_annotated_files = seg_dict[DB][1]["files"]
#        list_annotated_files = [file[:6] for file in list_annotated_files]
#        for file in os.listdir(folder):
#            if (file[:6] in list_annotated_files) & ((".tif" in file) or (".png" in file) or ("jpg" in file)) :
#                img_dict[DB]["img"].append(Image.open(os.path.join(folder,file)))
#                img_dict[DB]["files"].append(file[:-4])
#pkl.dump(img_dict, open(os.path.join(INTERRATER_FOLDER,'img_dict_double_annotation.pkl'), 'wb'))

img_dict = pkl.load(open(os.path.join(INTERRATER_FOLDER,'img_dict_double_annotation.pkl'), 'rb'))

## Create a dataframe with interrater measures for each database
#dict_interrater = {}
#for DB in DB_list :
#    print(DB)
#    dict_interrater[DB] = pd.DataFrame({"file_img" : img_dict[DB]["files"], 
#                   "file_annot1" : seg_dict[DB][0]["files"],
#                   "file_annot2" : seg_dict[DB][1]["files"]})
#    
#    # Add database name
#    dict_interrater[DB]["DB"]=DB
#    
#    # IoU
#    dict_interrater[DB]["IoU"]=[interrater_metrics.IoU(seg_dict[DB][0]["img"][i].reshape(-1),
#                    seg_dict[DB][1]["img"][i].reshape(-1)) for i in range(len(seg_dict[DB][0]["img"]))]
#    # Scaled IoU
#    dict_interrater[DB]["scaled_IoU"]=[interrater_metrics.scaled_IoU(seg_dict[DB][0]["img"][i].reshape(-1),
#                    seg_dict[DB][1]["img"][i].reshape(-1)) for i in range(len(seg_dict[DB][0]["img"]))]
#    
#    # Entropy
#    dict_interrater[DB]["entropy"]=[interrater_metrics.entropy(seg_dict[DB][0]["img"][i].reshape(-1),
#                    seg_dict[DB][1]["img"][i].reshape(-1)) for i in range(len(seg_dict[DB][0]["img"]))]
#    # Scaled Entropy
#    dict_interrater[DB]["scaled_entropy"]=[interrater_metrics.scaled_entropy(seg_dict[DB][0]["img"][i].reshape(-1),
#                    seg_dict[DB][1]["img"][i].reshape(-1)) for i in range(len(seg_dict[DB][0]["img"]))]
#
#dict_interrater["all"] = pd.concat([dict_interrater["aria"], dict_interrater["chasedb1"],dict_interrater["stare"]])
#dict_interrater["all"]    
#
#pkl.dump(dict_interrater, open(os.path.join(INTERRATER_FOLDER,'dict_interrater.pkl'), 'wb'))
dict_interrater = pkl.load(open(os.path.join(INTERRATER_FOLDER,'dict_interrater.pkl'), 'rb'))

metrics_list = ["IoU","scaled_IoU","entropy","scaled_entropy"]

for density in [True,False]:
    for metrics in metrics_list : 
        for DB in DB_list: 
            plt.hist(dict_interrater[DB][metrics],alpha=0.3, density=density, label=DB)
            plt.title(metrics)
            plt.legend()
        plt.savefig(str("interrater\\figures\\interrater"+metrics+"density"+str(density)+".png"))
        plt.show()
        














