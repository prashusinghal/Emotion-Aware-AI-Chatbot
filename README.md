# Emotion-Aware AI Chatbot 🤖

An AI system that combines **facial emotion recognition** and **natural language sentiment analysis** to generate intelligent responses.

##  Features

* Real-time facial emotion detection using CNN
* Sentiment analysis of user input text
* Emotion-aware chatbot response system
* Built using OpenCV, TensorFlow, and NLP techniques

##  Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* NLP (TextBlob)

##  How It Works

1. Captures face using webcam
2. Detects emotion (Happy, Sad, Angry, etc.)
3. Takes user text input
4. Analyzes sentiment
5. Generates a contextual response

##  Run the Project

```bash
pip install -r requirements.txt
python main.py
```

##  Files Included

* main.py → main logic
* emotion_model.hdf5 → trained CNN model
* sentiment_data.csv → sample dataset

##  Future Improvements

* Upgrade to transformer-based NLP (BERT)
* Deploy as web app (Streamlit)
* Add voice input/output

---
