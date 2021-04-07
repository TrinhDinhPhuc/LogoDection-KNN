import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import os
import sys
import pickle
from sklearn.neighbors import KNeighborsClassifier
from preprocessing_imgs import img_to_histogram
from pathlib import Path
from skimage import exposure, feature

    
if __name__ == "__main__":
    my_file = Path("hists.txt")
    if not my_file.is_file():
        print("Reading imgs...")
        img_to_histogram()

    # Get Paths
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    head, _ = os.path.split(ROOT_DIR)
    testPath = head + "/" + "mixLogo"

    print("Files created! Reading pickle files...")
    # Unpickling as the pickle file
    with open("hists.txt", "rb") as fp:
        hists = pickle.load(fp)
    with open("labels.txt", "rb") as fp:
        labels = pickle.load(fp)

    print("Read files! Training KNN model...")
    # Create model as Nearest Neighbors Classifier
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(hists, labels)

    # Check Test Images for Model
    for (imagePath) in glob.glob(testPath + "/*.*"):
        # Read Images
        image = cv.imread(imagePath)
        try:
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

            # Predict in model
            predict = model.predict(hist.reshape(1, -1))[0]

            # Make pictures default Height
            height, width = image.shape[:2]
            reWidth = int((300/height)*width)
            image = cv.resize(image, (reWidth, 300))

            # Write predicted label over the Image
            cv.putText(image, predict.title(), (10, 30), cv.FONT_HERSHEY_TRIPLEX, 1.2, (0 ,255, 0), 4)

            # Get Image name and show Image
            imageName = imagePath.split("/")[-1]
            plt.imshow(image)
            plt.show()
            cv.waitKey(0)
            # Close Image
            cv.destroyAllWindows()
        except cv.error:
            # If Image couldn't be Read
            print(imagePath)
            print("Test Image couldn't be read")