import cv2
import os
import face_recognition as fr
import numpy as np

# create path for images and name extraction
path = "/Users/umar/Downloads/6thSem/AI_Lab/images/"
images = []
names = []
myList = os.listdir(path)
print(myList)

# find the name of the person from image name and add images to a list

for imgNames in myList:
    curImg = cv2.imread(f"{path}/{imgNames}")
    images.append(curImg)
    names.append(os.path.splitext(imgNames)[0])

# find the face encodings of the images


def findEncodings(images):
    encodedlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = fr.face_encodings(img)[0]
        encodedlist.append(encode)
    return encodedlist


encodedListKnown = findEncodings(images)
cap = cv2.VideoCapture(0)

# open video capture and detect a face, find and compare its encodings by the distance, and if the distance is within the min range, show recognition
while True:
    _, webcam = cap.read()
    imgResized = cv2.resize(webcam, (0, 0), None, 0.25, 0.25)
    imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)

    faceCurFrame = fr.face_locations(imgResized)
    encodeFaceCurFrame = fr.face_encodings(imgResized, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeFaceCurFrame, faceCurFrame):
        matches = fr.compare_faces(encodedListKnown, encodeFace)
        faceDis = fr.face_distance(encodedListKnown, encodeFace)

        # argmin returns the indices of the minimum values along axis
        matchIndex = np.argmin(faceDis)
        # checks for match; if match found, shows name from the image name, and the face match percentage
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(webcam, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(webcam, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(
                webcam,
                name,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                1,
            )
            face_match_percentage = (1 - faceDis) * 100
            for i, face_distance in enumerate(faceDis):
                print(
                    "The test image has a distance of {:.2} from known image {} ".format(
                        face_distance, i
                    )
                )
                print(
                    "- comparing with a tolerance of 0.6 {}".format(face_distance < 0.6)
                )
                print(
                    "Face Match Percentage = ", np.round(face_match_percentage, 4)
                )  # upto 4 decimal places

    # allow exit from the webcam loop by escape sequence

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break
    cv2.imshow("webcam", webcam)
