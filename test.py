import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("data/drive/test/images/01_test.tif")

plt.imshow(img)
plt.show()