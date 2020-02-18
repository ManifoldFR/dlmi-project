import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2


img = Image.open("data\\stare\\im0001.ppm")
# img = Image.open("data\\drive\\test\\mask\\01_test_mask.gif")

# 
# img = Image.open("data\\drive\\test\\images\\01_test.tiff")
# img = cv2.imread("data\\drive\\test\\images\\01_test.tiff")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img)

plt.imshow(np.array(img))
plt.show()




