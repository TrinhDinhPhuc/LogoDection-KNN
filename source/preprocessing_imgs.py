from skimage import exposure, feature
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import os
import sys
import pickle

# Get Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
head, _ = os.path.split(ROOT_DIR)
trainingPath = head + "/" + "logos"

# Init Lists
hists = [] # histogram of Image
labels = [] # Label of Image

def img_to_histogram():
    for imagePath in glob.glob(trainingPath + "/*/*.*"):
        # get label from folder name
        label = imagePath.split("/")[-2]
        
        image = cv.imread(imagePath)
        try:
            # RGB to Gray
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # Calculate Low and Up value to extract Edges
            md = np.median(gray)
            sigma = 0.35
            low = int(max(0, (1.0 - sigma) * md))
            up = int(min(255, (1.0 + sigma) * md))
            # Create Edged Image from Gray Scale
            edged = cv.Canny(gray, low, up)
            plt.imread(edged)
            # extract only shape in image
            (x, y, w, h) = cv.boundingRect(edged) 
            logo = gray[y:y + h, x:x + w]
            logo = cv.resize(logo, (200, 100))

            # Calculate histogram
            hist = feature.hog(
                    logo, 
                    orientations=9, 
                    pixels_per_cell=(10, 10),cells_per_block=(2, 2),
                    transform_sqrt=True,
                    block_norm="L1"
                )
            ## Plot histogram
            # plt.imshow(np.expand_dims(hist,1))
            # plt.hist(hist, bins= 20)
            # plt.show()
            # print(hist.shape)
            
            # Add value into Lists
            hists.append(hist)
            labels.append(label)
        except cv.error:
            # If Image couldn't be Read
            print(imagePath)
            print("Training Image couldn't be read")

    # Pickling as a pickle file
    with open("hists.txt", "wb") as fp:  
        pickle.dump(hists, fp)

    with open("labels.txt", "wb") as fp: 
        pickle.dump(labels, fp)

