# import libraries
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2 as cv
import os
import json
import pprint

# read predictive images
prefix = "dataset/img"
# finds number of items in folder dataset
length = len(os.listdir("dataset"))
image_type = ".jpg"
image_path = (
    prefix + input("Please enter images from 1 to " + str(length) + " : ") + image_type
)

img = cv.imread(image_path)
# img read in brg order of sequence
print(image_type)
print("image matrices are: \n")
print(img)

# Model Prediction
# no need to distribute or split dataset since we are using existing model deepface

# brg to rgb in python
plt.imshow(img[:, :, ::-1])
attributes = ["gender", "race", "age", "emotion"]
result = DeepFace.analyze(img, attributes)
plt.title("Hello " + result["gender"])
pprint.pprint(result)
