from skimage import exposure, feature
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import os
import sys


# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)
trainingPath = head + "/" + "logos"

# Init Lists
hists = [] # histogram of Image
labels = [] # Label of Image

# for imagePath in glob.glob(trainingPath + "/*/*.*"):
#     # get label from folder name
#     label = imagePath.split("/")[-2]
#     print(label)


image = cv.imread("/media/harry/새 볼륨/CODE/Logo detection/logoDetection/logos/ford/00002.png")
plt.imshow(image)
plt.show()

# Convert to Gray and Resize
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
logo = cv.resize(gray, (200, 100))

# Calculate Histogram of Test Image
hist =  feature.hog(
            logo,
            orientations=9,
            pixels_per_cell=(10, 10),
            cells_per_block=(2, 2),
            transform_sqrt=True,
            block_norm="L1"
    )

# Make pictures default Height
height, width = image.shape[:2]
reWidth = int((300/height)*width)
image = cv.resize(image, (reWidth, 300))
# cv.imshow("ford Grouthtruth", image)
# plt.show()

# Write predicted label over the Image
image = cv.putText(image, "Ford Prediction", (10, 30), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0 ,255, 0), 4)
# Displaying the image
plt.imshow(image) 
plt.show()

# Get Image name and show Image
# cv.imshow("ford Grouthtruth", image)

sys.exit(1)



image = cv.imread("/media/harry/새 볼륨/CODE/Logo detection/logoDetection/logos/ford/00002.png")
plt.imshow(image)
plt.show()
# RGB to Gray
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Calculate Low and Up value to extract Edges
md = np.median(gray)
sigma = 0.35
low = int(max(0, (1.0 - sigma) * md))
up = int(min(255, (1.0 + sigma) * md))
# Create Edged Image from Gray Scale
edged = cv.Canny(gray, low, up)
plt.imshow(edged)
plt.show()
# extract only shape in image
(x, y, w, h) = cv.boundingRect(edged)
logo = gray[y:y + h, x:x + w]
logo = cv.resize(logo, (200, 100))
plt.imshow(logo)
plt.show()
# Calculate histogram
fd, hog_image = feature.hog(
        logo, 
        orientations=9, 
        pixels_per_cell=(10, 10),cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )


# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()