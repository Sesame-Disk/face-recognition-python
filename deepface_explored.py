from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import imshowpair
import pprint

images_path = "deepface_dataset/"
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
]

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
img1 = plt.imread("/deepface_datset/taylor1.jpg")
img2 = plt.imread("/deepface_datset/taylor2.jpg")
verification = DeepFace.verify(
    img1_path=img1, img2_path="/deepface_datset/taylor2.jpg", model_name="Dlib"
)
imshowpair.imshowpair(img1, img2)
print(verification)

# df = DeepFace.find(img_path="taylor1.jpg", db_path="deepface_dataset/")
# demography = DeepFace.analyze(img_path="img4.jpg", detector_backend=backends[4])
# DeepFace.stream(db_path="deepface_dataset")
# face = DeepFace.detectFace(
#     img_path="img9.jpg", target_size=(224, 224), detector_backend=backends[4]
# )
# embedding = DeepFace.represent(img_path="img3.jpg", model_name="Facenet")

df = DeepFace.find(img_path="/content/taylor1.jpg", db_path="/content")
print(df)

analysis = DeepFace.analyze(img_path="/content/img2.jpg", detector_backend=backends[4])
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(analysis)

DeepFace.stream(db_path="/content")
face = DeepFace.detectFace(
    img_path="/content/img9.jpg", target_size=(224, 224), detector_backend=backends[4]
)
print(face)
face = DeepFace.detectFace(
    img_path="/content/taylor2.jpg",
    target_size=(224, 224),
    detector_backend=backends[4],
)
print(face)
embedding = DeepFace.represent(img_path="img2.jpg", model_name="Facenet")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(embedding)
