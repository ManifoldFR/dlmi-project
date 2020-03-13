

import os
import numpy as np
import pandas as pd
from PIL import Image


######################################################################
### STARE DATABASE 
######################################################################

# PROJECT_FOLDER="C:\\Users\\Philo\\Documents\\3A -- MVA\\DL for medical imaging\\retine\\dlmi-project\\"
# DATA_FOLDER = os.path.join(PROJECT_FOLDER,"data", "stare")
DATA_FOLDER = os.path.join("data", "stare")
print(DATA_FOLDER)
diagnose=pd.read_excel(os.path.join(DATA_FOLDER,"diagnose.xls"))


# create one column only for diagnosis
diagnose[diagnose.columns[3]][diagnose[diagnose.columns[4]].isnull()==False]=diagnose[diagnose.columns[3]][diagnose[diagnose.columns[4]].isnull()==False].str.cat(diagnose[diagnose.columns[4]][diagnose[diagnose.columns[4]].isnull()==False],sep =" ")
diagnose=diagnose.drop(diagnose.columns[4],axis=1)
diagnose[diagnose.columns[3]][diagnose[diagnose.columns[2]].isnull()==False]=diagnose[diagnose.columns[3]][diagnose[diagnose.columns[2]].isnull()==False].str.cat(diagnose[diagnose.columns[2]][diagnose[diagnose.columns[2]].isnull()==False],sep =" ")
diagnose=diagnose.drop(diagnose.columns[2],axis=1)
diagnose.columns=["img","code","description"]
diagnose["description"][diagnose["description"].isnull()]=""
diagnose.to_excel(os.path.join(DATA_FOLDER,"diagnose_clean.xls"))

# Distribution over the entire dataset
print(sum(diagnose["code"]==0))
print(sum(diagnose["description"].str.contains("Normal") | diagnose["description"].str.contains("normal")))

# Distribution over the trainset only
# labels trainset 
labeled_index=os.listdir(os.path.join(DATA_FOLDER,'labels','labels_ah'))
labeled_index=[int(labeled_index[i][2:6].lstrip("0"))-1 for i in range(len(labeled_index))]
display(diagnose.iloc[labeled_index])
print(sum(diagnose.iloc[labeled_index]["code"]==0))
print(sum(diagnose.iloc[labeled_index]["description"].str.contains("Normal") | diagnose["description"].str.contains("normal")))



## Conversion to png format (lossless)
for im_name in os.listdir(os.path.join(DATA_FOLDER,"images")):
#    print(im_name)
    im = Image.open(os.path.join(DATA_FOLDER, "images", im_name))
    im.save(os.path.join(DATA_FOLDER, "images", str(im_name[:-4]+".png")),"PNG",quality=100)

for im_name in os.listdir(os.path.join(DATA_FOLDER,"labels","labels_ah")):
    print(im_name)
    im = Image.open(os.path.join(DATA_FOLDER, "labels","labels_ah", im_name))
    im.save(os.path.join(DATA_FOLDER, "annotation 1", str(im_name[:-4]+".png")),"PNG",quality=100)

for im_name in os.listdir(os.path.join(DATA_FOLDER,"labels","labels_vk")):
    print(im_name)
    im = Image.open(os.path.join(DATA_FOLDER, "labels","labels_vk", im_name))
    im.save(os.path.join(DATA_FOLDER, "annotation 2", str(im_name[:-4]+".png")),"PNG",quality=100)






