import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from transformers import pipeline
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Emotion AI Chatbot", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background-color: #0E1117;
}
.chat-user {
    background-color: #1F2937;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}
.chat-bot {
    background-color: #111827;
    padding: 10px;
    border-radius: 12px;
    margin-bottom: 8px;
}
.emotion-box {
    font-size: 20px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODELS ----------------
sentiment_model = pipeline("sentiment-analysis")
emotion_model = load_model("emotion_model.hdf5")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- SESSION ----------------
if "mode" not in st.session_state:
    st.session_state.mode = "idle"

if "emotion" not in st.session_state:
    st.session_state.emotion = "Neutral"

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- FUNCTIONS ----------------
def get_sentiment(text):
    result = sentiment_model(text)[0]
    return "Positive" if result['label'] == 'POSITIVE' else "Negative"

import random

def generate_response(face, sentiment, text):
    text = text.lower()

    history = " ".join([msg[1].lower() for msg in st.session_state.messages[-4:]])

    # ---------- GREETINGS ----------
    if any(word in text for word in ["hello", "hi", "hey"]):
        return random.choice([
            "Hey there! 👋 How are you feeling today?",
            "Hi! 😊 I'm here to chat with you!",
            "Hello! 👋 What's on your mind?"
        ])

    # ---------- CRITICAL ----------
    if any(word in text for word in ["kill", "die", "end my life"]):
        return random.choice([
            "I'm really glad you told me this 💔 You are not alone.",
            "Please reach out to someone you trust 🤍 You matter.",
            "I'm here with you ❤️ You deserve support."
        ])

    # ---------- SAD / DEPRESSED ----------
    if any(word in text for word in ["sad", "depressed", "lonely"]):
        return random.choice([
            "I'm really sorry you're feeling this way 💙",
            "You don’t have to go through this alone 🤍",
            "I'm here for you ❤️"
        ])

    # ---------- HAPPY ----------
    if "not happy" in text:
        return "I'm sorry you're not feeling happy. Want to talk about it? 💙"

    elif "happy" in text:
        return random.choice([
            "That's amazing! Keep smiling 😊✨",
            "Love that energy! 😄💫",
            "That makes me happy too! 🥰"
        ])

    # ---------- ANGRY ----------
    if "angry" in text or face == "Angry":
        return random.choice([
            "I can sense some frustration 😡 Take a deep breath 💪",
            "It's okay to feel angry sometimes. Want to talk about it?",
            "Let's slow things down and breathe together 🌿"
        ])

    # ---------- FEAR ----------
    if "scared" in text or "afraid" in text or face == "Fear":
        return random.choice([
            "That sounds scary 😨 I'm here with you.",
            "It's okay to feel afraid sometimes 🤍",
            "You're safe here. Want to share what's worrying you?"
        ])

    # ---------- SURPRISE ----------
    if face == "Surprise":
        return random.choice([
            "That looks surprising! 😲 What happened?",
            "Whoa! Something unexpected? Tell me more!",
            "That caught your attention 😮 What's going on?"
        ])

    # ---------- FACE-ONLY EMOTION ----------
    if face == "Happy":
        return random.choice([
            "You look happy 😄✨",
            "That smile suits you 😊",
            "Great vibes today 🌟"
        ])

    if face == "Sad":
        return random.choice([
            "I can sense you're feeling low 💙",
            "Take your time, I'm here 🤍",
            "You're not alone ❤️"
        ])

    if face == "Neutral":
        return random.choice([
            "How are you feeling today? 😊",
            "Anything on your mind?",
            "I'm listening 👀"
        ])

    # ---------- SENTIMENT ----------
    if sentiment == "Positive":
        return random.choice([
            "That's great to hear! 😄",
            "Nice! Keep going ✨",
            "Love the positivity! 💫"
        ])

    if sentiment == "Negative":
        return random.choice([
            "I understand… want to talk more? 🤍",
            "I'm here to listen 👀",
            "Tell me what's bothering you 💭"
        ])

    # ---------- MEMORY ----------
    if "sad" in history:
        return "I'm still here with you 💙 Do you want to talk more?"

    # ---------- DEFAULT ----------
    return random.choice([
        "Tell me more 👀",
        "I'm listening 💬",
        "Go on… 😊"
    ])

# ---------------- TITLE ----------------
st.markdown("## 🧠 Emotion-Aware AI Chatbot")
st.caption("Real-time Emotion Detection + Smart Chat System")

# ---------------- START ----------------
if st.session_state.mode == "idle":
    st.markdown("### 🚀 Ready to start?")
    st.button("▶ Start Emotion Detection", use_container_width=True,
              on_click=lambda: st.session_state.update(mode="camera"))

# ---------------- CAMERA ----------------
elif st.session_state.mode == "camera":
    st.subheader("📸 Detecting Emotion...")

    FRAME = st.empty()

    if "camera" not in st.session_state:
        st.session_state.camera = cv2.VideoCapture(0)

    camera = st.session_state.camera

    stop_button = st.button("⏹ Stop & Start Chat", use_container_width=True)

    if stop_button:
        st.session_state.mode = "chat"
        camera.release()
        del st.session_state.camera
        st.rerun()

    # 🔥 Smooth loop (but not blocking forever)
    for _ in range(150):   # runs ~3 seconds smooth video
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi / 255.0
            roi = np.reshape(roi, (1, 48, 48, 1))

            prediction = emotion_model.predict(roi, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]
            st.session_state.emotion = emotion

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        FRAME.image(frame, channels="BGR")
        time.sleep(0.03)

    # 🔁 Auto refresh after chunk
    st.rerun()

# ---------------- CHAT ----------------
elif st.session_state.mode == "chat":
    st.subheader("💬 Chat with Emotion AI")

    emotion = st.session_state.emotion

    # 🎨 Emotion color
    color = {
        "Happy": "🟢",
        "Sad": "🔵",
        "Angry": "🔴",
        "Neutral": "⚪"
    }.get(emotion, "⚪")

    st.markdown(f"### {color} Detected Emotion: `{emotion}`")

    # CHAT DISPLAY
    for sender, msg in st.session_state.messages:
        if sender == "You":
            st.markdown(f"<div class='chat-user'>👤 {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bot'>🤖 {msg}</div>", unsafe_allow_html=True)

    # INPUT
    user_input = st.chat_input("Type your message...")

    if user_input:
        sentiment = get_sentiment(user_input)

        # 🔥 typing feel
        with st.spinner("AI is typing..."):
            time.sleep(1)

        response = generate_response(emotion, sentiment, user_input)

        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))

        st.rerun()