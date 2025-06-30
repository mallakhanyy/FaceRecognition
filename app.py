import pickle
import numpy as np
import tkinter as tk
from tkinter import Label
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import cv2
from sklearn.preprocessing import LabelEncoder

# Load the trained model (saved using pickle earlier)
with open("face_recognition.pkl", "rb") as f:
    loaded = pickle.load(f)
    model = loaded[0] if isinstance(loaded, tuple) else loaded

data = np.load("faces_dataset.npz")
X_train, y_train = data["arr_0"], data["arr_1"]

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

embedder = FaceNet()
detector = MTCNN()

def recognize(frame, threshold=0.6):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)
    names_found = []

    for result in faces:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = rgb_frame[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160)) # resize for facenet
        face = face.astype('float32')
        sample = np.expand_dims(face, axis=0)
        embedding = embedder.embeddings(sample)

        probs = model.predict_proba(embedding)[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]
        name = label_encoder.inverse_transform([best_idx])[0]

        if confidence >= threshold:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (76, 175, 80), 2)
            cv2.putText(frame, f"{name}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (76, 175, 80), 2)
            names_found.append(name)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (244, 67, 54), 2)
            cv2.putText(frame, "Unrecognized", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (244, 67, 54), 2)
            names_found.append("Unrecognized")

    if not names_found:
        return "No face detected"
    return ", ".join(names_found)


# Keeps checking the camera and shows results
def update_gui():
    ret, frame = cap.read()
    if ret:
        result = recognize(frame)
        label_status.config(text=f"Current: {result}")
        cv2.imshow("Live Face Recognition", frame)
    root.after(1000, update_gui)

def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    root.destroy()

cap = cv2.VideoCapture(0)

root = tk.Tk()
root.title("Face Recognition")
root.geometry("450x250")
#Window icon
root.iconbitmap(r"E:\FaceRecognition\icon\artificial-intelligence.ico")
root.configure(bg="#F6DED8")

BG_COLOR = "#F6DED8"
TEXT_COLOR = "#D2665A"
TITLE_COLOR = "#B82132"
BTN_BG = "#F2B28C"
BTN_TEXT = "#ffffff"

label_title = Label(
    root, text="Face Recognition",
    font=("Helvetica", 20, "bold"), fg=TITLE_COLOR, bg=BG_COLOR
)
label_title.pack(pady=20)

label_status = Label(
    root, text="Waiting for face...",
    font=("Helvetica", 14), fg=TEXT_COLOR, bg=BG_COLOR
)
label_status.pack(pady=20)

btn_exit = tk.Button(
    root, text="Exit", font=("Helvetica", 12, "bold"),
    command=exit_app, bg=BTN_BG, fg=BTN_TEXT,
    padx=20, pady=5, relief="flat", borderwidth=0
)
btn_exit.pack(pady=20)

update_gui()
root.mainloop()