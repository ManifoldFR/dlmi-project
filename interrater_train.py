

import os 
from PIL import Image
from utils import interrater_metrics
import pickle as pkl

#PROJECT_FOLDER = os.getcwd()
PROJECT_FOLDER="C:\\Users\\Philo\\Documents\\3A -- MVA\\DL for medical imaging\\retine\\dlmi-project\\"

DB_list = ["aria","chasedb1","stare"]
nb_annot = 2

# Build dictionnary of annotations 
seg_dict = {}
for DB in DB_list : 
    DATA_FOLDER = os.path.join(PROJECT_FOLDER,"data", DB)
    seg_dict[DB]={}
    for i in range(nb_annot) :
        seg_dict[DB][i]={"img":[],"files":[]}
        folder = os.path.join(DATA_FOLDER,str("annotation "+str(i+1)))
        for file in os.listdir(folder):
            if (".tif" in file) or (".png" in file) :
                seg_dict[DB][i]["img"].append(Image.open(os.path.join(folder,file)))
                seg_dict[DB][i]["files"].append(file[:-4])
pkl.dump(seg_dict, open(os.path.join(PROJECT_FOLDER, "data", 'seg_dict_double_annotation.pkl'), 'wb'))


seg_dict = pkl.load(open(os.path.join(PROJECT_FOLDER, "data", 'seg_dict_double_annotation.pkl'), 'rb'))
