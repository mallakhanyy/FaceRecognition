from mtcnn.mtcnn import MTCNN
import cv2 as cv
import numpy as np
from numpy import savez_compressed, asarray
import os
import matplotlib.pyplot as plt
from PIL import Image
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import pickle

# Extract single face from image
detector = MTCNN()
def extract_face(filename, required_size=(160,160)):
  image = Image.open(filename).convert('RGB')
  pixels = asarray(image)
  results = detector.detect_faces(pixels)
  if len(results) == 0:
    return None
  x1, y1, width, height = results[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  face = pixels[y1:y2, x1:x2]
  image = Image.fromarray(face)
  image = image.resize(required_size)
  face_array = asarray(image)
  return face_array

folder = r"E:\FaceRecognition\Image\train\malak_hany"
i = 1

plt.figure(figsize=(12, 12))

for filename in os.listdir(folder):
    path = os.path.join(folder, filename)
    face = extract_face(path)

    if face is not None:
        print(i, face.shape)
        plt.subplot(10, 7, i)
        plt.axis('off')
        plt.imshow(face)
        i += 1
    else:
        print(f"No face detected in: {filename}")

    if i > 70:
        break

plt.tight_layout()
plt.show()

def load_faces(directory):
  faces = list()
  for filename in os.listdir(directory):
    path = directory + "/" + filename
    face = extract_face(path) # the previous function
    if face is not None:
      faces.append(face)
  return faces

def load_dataset(directory):
  x, y = list(), list()
  for subdir in os.listdir(directory):
    path = directory + "/" + subdir
    if not os.path.isdir(path):
      continue
    faces = load_faces(path) #the previous function
    if len(faces) == 0:
      continue
    labels = [subdir for _ in range(len(faces))]
    print(f"Loaded {len(faces)} examples for class: {subdir}")
    x.extend(faces)
    y.extend(labels)
  return asarray(x), asarray(y)

trainFolder = r"E:\FaceRecognition\Image\train"
testFolder = r"E:\FaceRecognition\Image\test"
trainX, trainY = load_dataset(trainFolder)
print(f"Train shape: {trainX.shape} <> {trainY.shape}")
testX, testY = load_dataset(testFolder)
print(f"Test shape: {testX.shape} <> {testY.shape}")
savez_compressed("faces_dataset.npz", trainX, trainY, testX, testY)

data = np.load("faces_dataset.npz")
trainX, trainY, testX, testY = data["arr_0"], data["arr_1"], data["arr_2"], data["arr_3"]
print(f"Loaded Train: {trainX.shape}, {trainY.shape}  Loaded Test: {testX.shape}, {testY.shape}")

# Extract face embeddings
embedder = FaceNet()
def get_embedding(face_pixels):
  face_pixels = face_pixels.astype("float32")
  samples = np.expand_dims(face_pixels, axis=0)
  yhat = embedder.embeddings(samples)
  return yhat[0]

def embedding(x):
  EMBEDDED_X = []
  for img in x:
      emb = get_embedding(img)
      if emb is not None:
          EMBEDDED_X.append(emb)
  EMBEDDED_X = np.asarray(EMBEDDED_X)
  return EMBEDDED_X

EMBEDDED_X = embedding(trainX)

# Normalize face embedding vector
inEncoder = Normalizer(norm="l2")
EMBEDDED_X = inEncoder.transform(EMBEDDED_X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(trainY)

model = SVC(kernel='linear', probability=True)
model.fit(EMBEDDED_X, y_encoded)

accuracy = model.score(EMBEDDED_X, y_encoded)
print(f"Training accuracy: {accuracy:.2%}")

# Save Model
with open('face_recognition.pkl', 'wb') as f:
    pickle.dump((model, label_encoder), f)

EMBEDDED_X_test = embedding(testX)
y_test_encoded = label_encoder.fit_transform(testY)

with open('face_recognition.pkl', 'rb') as file:
    model, label_encoder = pickle.load(file)

y_hat_test = model.predict(EMBEDDED_X_test)

acc = accuracy_score(y_test_encoded, y_hat_test)
print(f"Test Accuracy: {acc:.2%}\n")

print("Classification Report:\n", classification_report(y_test_encoded, y_hat_test, target_names=label_encoder.classes_))

print("Confusion Matrix:\n", confusion_matrix(y_test_encoded, y_hat_test))