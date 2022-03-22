import os
import cv2
import numpy as np
def loadImages(dataPath):
    """
    Load all Images in the folder and transfer a list of tuples.
    The first element is the numpy array of shape (m, n) representing the image.
    (remember to resize and convert the parking space images to 36 x 16 grayscale images.)
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    dataset = []
    for filename in os.listdir(dataPath+"/car"):
        image = cv2.cvtColor(cv2.resize(cv2.imread(dataPath+"/car/"+filename),(36,16)),cv2.COLOR_BGR2GRAY)
        dataset.append((np.asarray(image),1))
    for filename in os.listdir(dataPath+"/non-car"):
        image = cv2.cvtColor(cv2.resize(cv2.imread(dataPath+"/non-car/"+filename),(36,16)),cv2.COLOR_BGR2GRAY)
        dataset.append((np.asarray(image),0))
    # raise NotImplementedError("To be implemented")
    # End your code (Part 1)

    return dataset
