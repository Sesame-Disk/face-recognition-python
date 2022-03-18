#Dockerfile: Image, Container
#Dockerfile-- blueprint for building images
# image is template for running Container
# container is actual project code

#specify a baseimage in this case python
#so this pulls this image from dockerhub with this python
FROM python:3.9.7

#next thing to add face-recognition-python.py to comntainer
ADD face-recognition-python-attendance.py .

#get source destination
# . is current directory


#now install dependencies
RUN pip install cv2, face_recognition, numpy, datetime 

# specify the entry command when we start contaienr
CMD ["python", ".face-recognition-python-attendance.py"]

#be mindful of spacing after CAPS words
