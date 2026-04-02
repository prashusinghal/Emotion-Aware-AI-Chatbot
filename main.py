import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load CNN model
emotion_model = load_model("emotion_model.hdf5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load NLP dataset
data = pd.read_csv("sentiment_data.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["sentiment"]

model = LogisticRegression()
model.fit(X, y)

# Face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("Press 'q' to capture emotion")

face_emotion = "Neutral"

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = emotion_model.predict(roi)
        face_emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, face_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Text input
user_text = input("Enter your message: ")

text_vec = vectorizer.transform([user_text])
text_sentiment = model.predict(text_vec)[0]

# Decision logic
def decide_response(face, text):
    if face == "Sad" and text == "Negative":
        return "You seem upset. Take a break."
    if face == "Neutral" and text == "Negative":
        return "You seem stressed. Try relaxing."
    if face == "Happy" and text == "Positive":
        return "Great! Keep it up."
    return "Stay focused and calm."

response = decide_response(face_emotion, text_sentiment)

print("\n--- OUTPUT ---")
print("Face Emotion:", face_emotion)
print("Text Sentiment:", text_sentiment)
print("Response:", response)