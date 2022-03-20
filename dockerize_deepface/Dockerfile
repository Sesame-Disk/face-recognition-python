#Dockerfile: Image, Container
#Dockerfile-- blueprint for building images
# image is template for running Container
# container is actual project code

#specify a baseimage in this case python
#so this pulls this image from dockerhub with this python
FROM python:3.9.7

#next thing to add face-recognition-python.py to container
ADD deepface_explored.py .

#get source destination
# . is current directory


#now install dependencies
RUN pip install deepface, matplotlib

# specify the entry command when we start container
CMD ["python", ".deepface_explored.py"]

#be mindful of spacing after CAPS words
