import cv2
import os

# Create folder if not exists
if not os.path.exists("data"):
    os.makedirs("data")

def generate_dataset(img, id, img_id):
    path = f"Face-Detection/data/user.{id}.{img_id}.jpg"
    success = cv2.imwrite(path, img)
    print("Saved:", path, "->", success)

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
        coords = [x, y, w, h]
    return coords

def detect(img, faceCascade, eyeCascade, noseCascade, mouthCascade, img_id):
    color = {"blue":(255,0,0), "white":(255,255,255)}
    coords = draw_boundary(img, faceCascade, 1.1, 10, color['white'], "Face")
    
    if len(coords)==4:
        roi_img = img[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        user_name = 3
        if roi_img is not None and roi_img.size != 0:
            generate_dataset(roi_img, user_name, img_id)
        else:
            print("ROI is empty—not saving.")
    
    return img

faceCascade = cv2.CascadeClassifier(r"K:\Python_Programing\Face-Detection\haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)
img_id = 0

while True:
    ret, img = video_capture.read()

    if not ret or img is None:
        print("Camera not available.")
        continue

    img = detect(img, faceCascade, None, None, None, img_id)
    cv2.imshow("face detection", img)

    img_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
