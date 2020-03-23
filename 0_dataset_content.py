

import os
import numpy as np
import pandas as pd
from PIL import Image

# PROJECT_FOLDER="C:\\Users\\Philo\\Documents\\3A -- MVA\\DL for medical imaging\\retine\\dlmi-project\\"


######################################################################
### STARE DATABASE 
######################################################################

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


diagnose_train = diagnose[diagnose.index.isin(labeled_index)]
diagnose_train.index = range(len(diagnose_train))

stare_dict={"images" : os.listdir(os.path.join(DATA_FOLDER, "images")),
            "STAPLE" : os.listdir(os.path.join(DATA_FOLDER, "STAPLE")),
            "annot1" : os.listdir(os.path.join(DATA_FOLDER, "annotation 1")),
            "annot2" : os.listdir(os.path.join(DATA_FOLDER, "annotation 2"))}

stare_dict["images"] = [file for file in stare_dict["images"] if (".png" in file) and (int(file[2:6])-1 in labeled_index)]
stare_dict["annot1"] = [file for file in stare_dict["annot1"] if ".png" in file]
stare_dict["annot2"] = [file for file in stare_dict["annot2"] if ".png" in file]
stare_dict["disc_fovea_available"] = 0
stare_dict["disc_fovea"] = [""]*len(stare_dict["images"])


stare_df = pd.DataFrame(stare_dict)
stare_df = pd.concat([stare_df, diagnose_train], axis=1)
stare_df = stare_df.drop(["img"], axis=1)

stare_df.to_csv(os.path.join(DATA_FOLDER,"stare_df.csv"))


######################################################################
### ARIA DATABASE 
######################################################################

DATA_FOLDER = os.path.join("data", "aria")

aria_dict={"images" : os.listdir(os.path.join(DATA_FOLDER, "images")),
            "STAPLE" : os.listdir(os.path.join(DATA_FOLDER, "STAPLE")),
            "annot1" : os.listdir(os.path.join(DATA_FOLDER, "annotation 1")),
            "annot2" : os.listdir(os.path.join(DATA_FOLDER, "annotation 2"))}

aria_dict["annot1"] = [file for file in aria_dict["annot1"] if ".tif" in file]
aria_dict["annot2"] = [file for file in aria_dict["annot2"] if ".tif" in file]
aria_dict["healthy"] = [("aria_c_" in file)*1 for file in aria_dict["images"]]
aria_dict["diabetic"] = [("aria_d_" in file)*1 for file in aria_dict["images"]]
aria_dict["AMD"] = [("aria_a_" in file)*1 for file in aria_dict["images"]]
aria_dict["disc_fovea_available"] = np.array(aria_dict["diabetic"]) + np.array(aria_dict["healthy"])
aria_dict["disc_fovea"] = [""]*np.sum(aria_dict["AMD"])+os.listdir(os.path.join(DATA_FOLDER, "markupdiscfovea"))[:-1]

aria_df = pd.DataFrame(aria_dict)

aria_df.to_csv(os.path.join(DATA_FOLDER,"aria_df.csv"))


######################################################################
### CHASEDB DATABASE 
######################################################################

DATA_FOLDER = os.path.join("data", "chasedb1")

chase_dict={"images" : os.listdir(os.path.join(DATA_FOLDER, "images")),
            "STAPLE" : os.listdir(os.path.join(DATA_FOLDER, "STAPLE")),
            "annot1" : os.listdir(os.path.join(DATA_FOLDER, "annotation 1")),
            "annot2" : os.listdir(os.path.join(DATA_FOLDER, "annotation 2"))}

chase_dict["annot1"] = [file for file in chase_dict["annot1"] if ".png" in file]
chase_dict["annot2"] = [file for file in chase_dict["annot2"] if ".png" in file]
chase_dict["STAPLE"] = [file for file in chase_dict["STAPLE"] if ".png" in file]

chase_dict["left_eye"] = [("L.jpg" in file)*1 for file in chase_dict["images"]]
chase_dict["right_eye"] = [("R.jpg" in file)*1 for file in chase_dict["images"]]

chase_dict["disc_fovea_available"] = 0
chase_dict["disc_fovea"] = [""]*len(chase_dict["images"])

chase_df = pd.DataFrame(chase_dict)

chase_df.to_csv(os.path.join(DATA_FOLDER,"chase_df.csv"))




