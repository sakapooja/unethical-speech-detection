# 🛡️ Unethical Speech Detection System

A real-time speech monitoring system that listens through a microphone, transcribes speech with Google Speech Recognition, and flags potentially harmful language using a **hybrid** approach: a **priority keyword list** plus a **TF-IDF + Linear SVM** classifier. Harmful speech triggers an **audible beep**, and every transcript is **logged to a CSV file**.

## 📌 Problem Statement & Objectives

Abusive or unethical language contributes to hostile environments, and manual monitoring does not scale. This project aims to:

- Capture live speech and convert it to text automatically.
- Classify each transcript as `HARMFUL` or `SAFE` using machine learning and a keyword rule list.
- Alert immediately (sound) when harmful speech is detected.
- Keep a timestamped record of all detections for later review.

## ✨ Features

- **Continuous monitoring** – listens in a loop; waits for 1.8 s of silence before ending a phrase (max 12 s per phrase).
- **Ambient noise calibration** – 0.5 s calibration before each listen; dynamic energy threshold.
- **Speech-to-text** – Google Web Speech API via the `SpeechRecognition` library.
- **Hybrid detection** – flagged as harmful if the SVM predicts harmful **or** a priority word/phrase is found (whole-word match).
- **TF-IDF features** – unigrams + bigrams, English stop words removed, max 5,000 features.
- **Audible alert** – 1000 Hz beep for 600 ms (Windows `winsound`).
- **Incident logging** – appends `Time, Transcript, Category` rows to `incident_logs.csv`.

## 🧠 Technologies Used

| Component | Technology |
|---|---|
| Language | Python 3.11 – 3.13 |
| Speech capture / STT | SpeechRecognition, PyAudio, Google Speech Recognition |
| Features | scikit-learn `TfidfVectorizer` |
| Classifier | scikit-learn `LinearSVC` |
| Data handling | pandas (training only) |
| Model storage | pickle |
| Alerts | Windows `winsound` |
| Logging | CSV |

## 📁 Project Structure

```text
unethical-speech-detection/
├── main.py                    # Live monitoring: mic → STT → keyword + SVM check → beep → CSV log
├── train_model.py             # Downloads dataset, trains TF-IDF + LinearSVC, saves *_final.pkl
├── svm_model_final.pkl        # Trained LinearSVC (USED by main.py)
├── tfidf_final.pkl            # Fitted TF-IDF vectorizer (USED by main.py)
├── svm_model.pkl              # Earlier SVM model (not used by main.py)
├── tfidf_vectorizer.pkl       # Vectorizer paired with svm_model.pkl (not used by main.py)
├── svm_model_big.pkl          # Earlier TF-IDF + SVM scikit-learn Pipeline (not used by main.py)
├── sample_incident_logs.csv   # Example of the log format (no real data)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

`incident_logs.csv` is created automatically at runtime and is excluded from Git because it may contain real transcripts.

## 🗂️ Model Files

| File | Purpose |
|---|---|
| `svm_model_final.pkl` | Linear SVM that outputs `1` (harmful) or `0` (safe) |
| `tfidf_final.pkl` | Converts a transcript into the TF-IDF vector the SVM expects |
| `svm_model.pkl`, `tfidf_vectorizer.pkl`, `svm_model_big.pkl` | Earlier experiment models kept for reference |

The models were saved with **scikit-learn 1.8.0**, so that exact version is pinned in `requirements.txt`. Using another version may cause load errors or warnings.

## ⚙️ How It Works

1. **Speech detection** – `main.py` opens the default microphone, calibrates for background noise, and records a phrase.
2. **Transcription** – audio is sent to Google Speech Recognition and the text is lower-cased.
3. **Classification**
   - *Keyword check:* the text is searched for any entry in `PRIORITY_HARMFUL_WORDS` (whole words/phrases).
   - *ML check:* the text is transformed by `tfidf_final.pkl` and classified by `svm_model_final.pkl`.
   - If **either** check is positive → `HARMFUL`, otherwise `SAFE`.
4. **Alert** – on `HARMFUL`, a 1000 Hz / 600 ms beep plays.
5. **Logging** – a row is appended to `incident_logs.csv`:

| Column | Example | Description |
|---|---|---|
| `Time` | `10:15:41` | Local time (HH:MM:SS) |
| `Transcript` | `you are an idiot` | Recognized text |
| `Category` | `HARMFUL` / `SAFE` | Final decision |

**Training (`train_model.py`)** downloads the [Davidson et al. hate speech & offensive language dataset](https://github.com/t-davidson/hate-speech-and-offensive-language), maps classes 0/1 → harmful and 2 → safe, balances up to 10,000 samples per class, fits the TF-IDF vectorizer and a `LinearSVC(C=1.0, class_weight='balanced')`, and saves `svm_model_final.pkl` and `tfidf_final.pkl`. The printed accuracy is **training accuracy** (no held-out test set).

## 🚀 Installation (Windows)

**Prerequisites:** Windows 10/11, Python 3.11–3.13 ([python.org](https://www.python.org/downloads/), tick “Add python.exe to PATH”), Git, a microphone, and an internet connection.

```powershell
git clone https://github.com/sakapooja/unethical-speech-detection.git
cd unethical-speech-detection

py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> If PowerShell blocks activation, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## ▶️ Running the Project

```powershell
venv\Scripts\activate
python main.py
```

Speak into the microphone; results print to the console and are appended to `incident_logs.csv`. Press **Ctrl + C** to stop.

**Optional – retrain the model** (downloads the dataset and **overwrites** `svm_model_final.pkl` and `tfidf_final.pkl`; back them up first):

```powershell
python train_model.py
```

### Quick offline check (no microphone needed)

```powershell
python -c "import pickle; m=pickle.load(open('svm_model_final.pkl','rb')); v=pickle.load(open('tfidf_final.pkl','rb')); print(m.predict(v.transform(['you are stupid','have a nice day'])))"
```

Expected output: `[1 0]`

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `Error: Could not load model files` | Run from the project folder (the `.pkl` files must be in the current directory). |
| `InconsistentVersionWarning` / unpickling error | Install `scikit-learn==1.8.0` (`pip install -r requirements.txt`). |
| `pip install PyAudio` fails | Use Python 3.11–3.13 (prebuilt Windows wheels), then `pip install --upgrade pip` and retry. |
| `Could not find PyAudio` / no microphone | Check Windows *Settings → Privacy → Microphone* and set the correct default input device. |
| `Internet connection issue (Google API)` | Google Speech Recognition needs internet access; check network/firewall. |
| `Background noise ignored.` repeatedly | Speak closer to the mic; reduce background noise. |
| `ModuleNotFoundError: winsound` | You are not on Windows; the beep uses a Windows-only module. |

## ⚠️ Limitations

- Requires internet for transcription; audio is sent to Google’s service.
- Windows-only alert (`winsound`).
- English-only training data; tweet-style dataset may not match spoken language.
- Keyword list can flag non-abusive uses (e.g., “hit”, “kill time”, “bad”); the SVM can miss context, sarcasm and indirect abuse.
- Evaluation reports training accuracy only.
- Logs record time only (no date).
- Monitoring speech has privacy implications — obtain consent before use.

## 🔮 Future Enhancements

- Proper train/validation/test split with precision, recall, F1 and a confusion matrix.
- Date + time in logs and a dashboard to review incidents.
- Offline speech recognition (e.g., Vosk/Whisper) and cross-platform alerts.
- Configurable keyword list and thresholds via a config file.
- Multilingual / code-mixed speech support.

## 👥 Credits

Originally developed as a team project and first published at [Harini-adusumalli/unethical-speech-detection](https://github.com/Harini-adusumalli/unethical-speech-detection). Republished with permission by Sai Spandana.

Dataset: Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language.* ICWSM.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
