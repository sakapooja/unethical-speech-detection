import speech_recognition as sr
import pickle
import re
import csv
import os
import winsound
from datetime import datetime

# --- CONFIGURATION ---
# Add any words you want the system to catch instantly
PRIORITY_HARMFUL_WORDS = ["useless", "idiot", "shut up", "stupid", "kill", "hate", "loser", "hit", "ugly", "moron", "dumb", "fool", "suck", "trash", "not capable", "worthless", "pathetic", "disgusting", "nonsense", "garbage", "awful", "terrible", "horrible", "bad", "worst", "screw you", "damn", "bastard", "jerk", "asshole", "dickhead", "son of a bitch", "motherfucker", "crap", "bullshit", "piss off", "screw you", "fucking", "fucker", "ass", "bitch","shameless", "disgrace", "idiotic", "imbecile", "douchebag", "scumbag", "twat","fit for nothing","good for nothing","piece of shit"]

# 1. Load AI Brain (Model and Vectorizer)
try:
    with open('svm_model_final.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_final.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("AI Brain Loaded. System Ready for Review-2.")
except Exception as e:
    print(f"Error: Could not load model files. Error: {e}")
    print("Make sure you run 'train_model.py' first!")
    exit()

# 2. Logger Module (Output Generation)
def log_incident(text, status):
    file_exists = os.path.isfile('incident_logs.csv')
    with open('incident_logs.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Time', 'Transcript', 'Category'])
        # Logs the specific time, what was said, and if it was harmful or safe
        writer.writerow([datetime.now().strftime("%H:%M:%S"), text, status])

# 3. Main Execution Function
def run_monitor():
    rec = sr.Recognizer()
    
    # --- LIVE DETECTION SETTINGS ---
    # We increase this so it doesn't stop recording while you are still talking
    rec.pause_threshold = 1.8  # Wait 1.8 seconds of silence before finishing
    rec.dynamic_energy_threshold = True 
    
    with sr.Microphone() as source:
        print("\n--- [SYSTEM MONITORING ACTIVE] ---")
        # Quick calibration to handle room noise
        rec.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening for speech...")
        
        try:
            # Capture audio (no timeout so it keeps listening until speech starts)
            audio = rec.listen(source, timeout=None, phrase_time_limit=12)
            
            # Transcription Module
            print("Analyzing...")
            text = rec.recognize_google(audio).lower()
            print(f"Heard: {text}")

            # Hybrid Check: Keyword List + SVM AI Model
            # Whole-word/phrase match, so e.g. "ass" does not match "class" or "assignment"
            is_priority = any(re.search(r"\b" + re.escape(word) + r"\b", text) for word in PRIORITY_HARMFUL_WORDS)
            prediction = model.predict(vectorizer.transform([text]))[0]

            # Decision Logic
            if prediction == 1 or is_priority:
                print("⚠️  ALERT: HARMFUL SPEECH DETECTED!")
                
                # Sound Alert (Frequency: 1000Hz, Duration: 600ms)
                winsound.Beep(1000, 600) 
                
                log_incident(text, "HARMFUL")
            else:
                print("✅ STATUS: Safe Speech")
                log_incident(text, "SAFE")

        except sr.UnknownValueError:
            # This triggers if it hears noise but can't find words
            print("Status: Background noise ignored.")
        except sr.RequestError:
            print("Status: Internet connection issue (Google API).")
        except Exception as e:
            print(f"Status: {e}")

if __name__ == "__main__":
    print("Project: IoT-Enabled AI Speech Monitoring System")
    print("Starting continuous loop...")
    while True:
        run_monitor()