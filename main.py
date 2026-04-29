import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -------------------- LOAD EMOTION MODEL --------------------
emotion_model = load_model("emotion_model.hdf5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# -------------------- LOAD NLP MODEL --------------------
data = pd.read_csv("sentiment_data.csv")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["text"])
y = data["sentiment"]

nlp_model = LogisticRegression()
nlp_model.fit(X, y)

# -------------------- FUNCTIONS --------------------

def get_sentiment(text):
    text_vec = vectorizer.transform([text])
    return nlp_model.predict(text_vec)[0]

def generate_response(face, sentiment):
    if face == "Happy" and sentiment == "Positive":
        return "That's awesome! Keep smiling 😊"

    elif face == "Sad":
        return "I'm here for you. Things will get better ❤️"

    elif face == "Angry":
        return "Take a deep breath. Stay calm."

    elif sentiment == "Negative":
        return "I understand. Try to stay positive."

    else:
        return "Tell me more about it."

# -------------------- FACE DETECTION --------------------

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

print("📸 Press 'q' to capture emotion and start chatbot")

face_emotion = "Neutral"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 48, 48, 1))

        prediction = emotion_model.predict(roi, verbose=0)
        face_emotion = emotion_labels[np.argmax(prediction)]

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, face_emotion, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------- CHATBOT LOOP --------------------

print("\n🤖 Chatbot Started (type 'exit' to stop)\n")

chat_history = []

while True:
    user_text = input("You: ")

    if user_text.lower() == "exit":
        print("👋 Chat ended.")
        break

    text_sentiment = get_sentiment(user_text)

    response = generate_response(face_emotion, text_sentiment)

    chat_history.append(("You", user_text))
    chat_history.append(("Bot", response))

    print("\n--- OUTPUT ---")
    print("Face Emotion:", face_emotion)
    print("Text Sentiment:", text_sentiment)
    print("Bot:", response)
