from deepface import DeepFace
import matplotlib.pyplot as plt

images_path = "/Users/umar/Downloads/6thSem/AI_Lab/images/"
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


def faceVer(img1, img2, model_name):
    verification = DeepFace.verify(
        img1_path=images_path + img1,
        img1_path=images_path + img2,
        model_name=model_name,
    )
    plt.imshow(img1[:, :, ::-1])
    plt.imshow(img2[:, :, ::-1])
    plt.show()
    return verification


# def faceAna():


f = faceVer("umar_1.jpg", "umar_2.jpg", "Facenet512",)
df = DeepFace.find(
    img_path="umar.jpg", db_path="/Users/umar/Downloads/6thSem/AI_Lab/images/"
)
demography = DeepFace.analyze(img_path="img4.jpg", detector_backend=backends[4])
DeepFace.stream(db_path="C:/User/Sefik/Desktop/database")
face = DeepFace.detectFace(
    img_path="img.jpg", target_size=(224, 224), detector_backend=backends[4]
)
embedding = DeepFace.represent(img_path="img.jpg", model_name="Facenet")

