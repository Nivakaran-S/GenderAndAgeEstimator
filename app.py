import cv2
import os 
import time
import numpy as np 
import argparse 
from model import WideResNet

class FaceCV(object):
    """
    Singleton class for face recognition task
    """
    CASE_PATH = "./pretrained_models/haarcascade_frontalface_alt.xml"
    WRN_WEIGHTS_PATH = "./pretrained_models/weights.18-4.06.hdf5"

    def __new__(cls, weight_file=None, depth=16, width=8, face_size=64):
        if not hasattr(cls, 'instance'):
            cls.instance = super(FaceCV, cls).__new__(cls)
        return cls.instance

    def __init__(self, depth=16, width=8, face_size=64):
        self.face_size = face_size
        self.model = WideResNet(face_size, depth=depth, k=width)()
        if not os.path.exists(self.WRN_WEIGHTS_PATH):
            raise FileNotFoundError(f"Weight file not found at: {self.WRN_WEIGHTS_PATH}")
        self.model.load_weights(self.WRN_WEIGHTS_PATH)

    @classmethod
    def draw_label(cls, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def crop_face(self, imgarray, section, margin=40, size=64):
        img_h, img_w, _ = imgarray.shape
        if section is None:
            section = [0, 0, img_w, img_h]
        (x, y, w, h) = section
        margin = int(min(w, h) * margin / 100)
        x_a = max(x - margin, 0)
        y_a = max(y - margin, 0)
        x_b = min(x + w + margin, img_w)
        y_b = min(y + h + margin, img_h)
        cropped = imgarray[y_a:y_b, x_a:x_b]
        resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
        return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)
        video_capture = cv2.VideoCapture(0)

        while True:
            if not video_capture.isOpened():
                time.sleep(5)
                continue

            ret, frame = video_capture.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=10,
                minSize=(self.face_size, self.face_size)
            )

            if len(faces) > 0:
                face_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                for i, face in enumerate(faces):
                    face_img, cropped = self.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    face_imgs[i, :, :, :] = face_img

                results = self.model.predict(face_imgs)
                predicted_genders = results[0]
                ages = np.arange(0, 101).reshape(101, 1)
                predicted_ages = results[1].dot(ages).flatten()

                for i, face in enumerate(faces):
                    age = int(predicted_ages[i])
                    gender_prob = predicted_genders[i][0]
                    gender = "Female" if gender_prob > 0.5 else "Male"
                    label = f"{age}, {gender}"
                    print(age, gender_prob)

                    folder = None
                    if gender_prob < 0.5 and 25 < age < 30:
                        folder = "25-30"
                    elif gender_prob < 0.5 and 30 < age < 35:
                        folder = "30-35"
                    elif gender_prob > 0.5 and 25 < age < 30:
                        folder = "F25-30"

                    if folder and os.path.exists(folder):
                        filename = random.choice(os.listdir(folder))
                        filepath = os.path.join(folder, filename)
                        cap = cv2.VideoCapture(filepath)

                        while cap.isOpened():
                            ret, clip_frame = cap.read()
                            if ret:
                                cv2.imshow('Frame', clip_frame)
                                if cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                            else:
                                break
                        cap.release()
                        cv2.destroyAllWindows()

                    self.draw_label(frame, (face[0], face[1]), label)
            else:
                print("No faces")

            cv2.imshow("Keras Faces", frame)
            if cv2.waitKey(5) == 27:  # ESC to break
                break

        video_capture.release()
        cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser(description="Detects faces from webcam and estimates age/gender.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--depth", type=int, default=16, help="Depth of network")
    parser.add_argument("--width", type=int, default=8, help="Width of network")
    return parser.parse_args()

def main():
    args = get_args()
    face = FaceCV(depth=args.depth, width=args.width)
    face.detect_face()

if __name__ == "__main__":
    main()
