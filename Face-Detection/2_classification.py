import numpy as np
from PIL import Image
import os, cv2

def train_classifer(data_dir):
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []

    print("Found files:", path)

    for image in path:
        try:
            img = Image.open(image).convert('L')
        except:
            print("Cannot open file:", image)
            continue

        imageNp = np.array(img, 'uint8')

        try:
            id = int(os.path.split(image)[1].split(".")[1])
        except:
            print("Filename format incorrect:", image)
            continue

        faces.append(imageNp)
        ids.append(id)

    if len(faces) < 2:
        print("❌ Not enough training images!")
        print("Faces found:", len(faces))
        return

    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("Face-Detection\classifier.xml")

    print("✅ Training completed successfully!")

train_classifer("Face-Detection\data")
