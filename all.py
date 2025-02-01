import cv2
import whisper
import pyttsx3
import requests
import time
import sounddevice as sd
import numpy as np
import wave
from ultralytics import YOLO
from deepface import DeepFace

# Load the object detection model on GPU (for other tasks, if needed)
detection_model = YOLO("yolov8n.pt").to("cuda")

# Initialize the face detector using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize text-to-speech (TTS)
engine = pyttsx3.init()

# Load the Whisper model on CPU (to avoid GPU memory issues)
whisper_model = whisper.load_model("small", device="cpu")

def speak(text):
    """Converts text to speech."""
    engine.say(text)
    engine.runAndWait()

def record_audio(filename, duration=5, fs=44100):
    """Records audio from the microphone and saves it as a WAV file."""
    print("Recording audio, please speak...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(fs)
        wf.writeframes(recording.tobytes())
    print("Recording finished.")

def listen():
    """Converts speech to text using Whisper."""
    audio_path = "input.wav"
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def chat_with_gpt(prompt):
    """
    Sends the query to Ollama using the 'mistral' model and returns only the 'response' field.
    Adjust the endpoint URL if needed.
    """
    data = {
        "model": "mistral",
        "stream": False,
        "prompt": prompt
    }
    url = "http://localhost:11434/api/generate"  # This endpoint works based on your tests

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Raise an exception if the response is not 200 OK
        json_resp = response.json()
        return json_resp.get("response", "I didn't understand the question.")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return "I'm sorry, I cannot respond at the moment."

def analyze_face(frame):
    """
    Analyzes the face using DeepFace and returns a description.
    Assumes 'frame' is an image in BGR format (as from cv2).
    """
    # Optionally, you could crop the detected face region. Here we use the full frame.
    # Convert from BGR to RGB (DeepFace works better in RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        analysis = DeepFace.analyze(rgb_frame, actions=['age', 'gender', 'emotion', 'race'], enforce_detection=False)
        # Extract some relevant attributes:
        gender = analysis.get('gender', 'unknown')
        age = analysis.get('age', 'unknown')
        dominant_emotion = analysis.get('dominant_emotion', 'neutral')
        dominant_race = analysis.get('dominant_race', 'unknown')
        description = (f"I see a {dominant_race} {gender} around {age} years old, "
                       f"and you seem {dominant_emotion}.")
        return description
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return "I couldn't analyze your face properly."

# Configure the camera (use index 2 for the Logitech camera; change to 0 if using the built-in camera)
cap = cv2.VideoCapture(2)

# Variables to control the interval between conversations
last_conversation_time = 0
conversation_interval = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("ChatGPT Vision", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # If at least one face is detected and enough time has passed, start a conversation
    if len(faces) > 0 and time.time() - last_conversation_time > conversation_interval:
        last_conversation_time = time.time()  # Update the time of the last conversation

        # Start the conversation
        speak("Hello, how are you?")
        record_audio("input.wav", duration=5)
        user_text = listen()
        print(f"You: {user_text}")

        # If the user asks to describe their face, analyze the current frame
        if "describe my face" in user_text.lower():
            description = analyze_face(frame)
            response_text = description
        else:
            response_text = chat_with_gpt(user_text)

        print(f"ChatGPT: {response_text}")
        speak(response_text)

cap.release()
cv2.destroyAllWindows()
