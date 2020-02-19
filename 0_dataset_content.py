import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

DATA_FOLDER = os.path.join("data", "drive", "test", "images")
print(DATA_FOLDER)

img = cv2.imread(os.path.join(DATA_FOLDER, "01_test.tif"),
                 cv2.IMREAD_UNCHANGED)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img)

plt.imshow(np.array(img))
plt.show()




