import streamlit as st
import librosa
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Speaker Recognition System",
    page_icon="🎙️",
    layout="centered"
)

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    # Check if they are in the 'saved_model' folder (local) or in the root folder (GitHub)
    if os.path.exists(os.path.join("saved_model", "speaker_svm_model.joblib")):
        model_path = os.path.join("saved_model", "speaker_svm_model.joblib")
        le_path = os.path.join("saved_model", "label_encoder.joblib")
    else:
        model_path = "speaker_svm_model.joblib"
        le_path = "label_encoder.joblib"
    
    if os.path.exists(model_path) and os.path.exists(le_path):
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        return model, le
    else:
        st.error(f"Model files not found! Looking for: `{model_path}` and `{le_path}`")
        
        # Debugging information to help find where the files actually are
        st.warning("### Debug Information")
        st.write(f"**Current Working Directory:** `{os.getcwd()}`")
        st.write("**Files in Current Directory:**", os.listdir("."))
        
        if os.path.exists("saved_model"):
            st.write("**Files in `saved_model` folder:**", os.listdir("saved_model"))
        else:
            st.write("❌ The `saved_model` folder does not exist in the current directory.")
            
        return None, None

model, le = load_models()

# --- FEATURE EXTRACTION ---
def extract_mfcc_mean(audio_path_or_bytes, fs=16000, n_mfcc=20):
    # Load audio using librosa
    audio, sr = librosa.load(audio_path_or_bytes, sr=fs)
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=fs, n_mfcc=n_mfcc)
    # Get mean of each coefficient
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

# --- UI FRONTEND ---
st.title("🎙️ Speaker Recognition System")
st.markdown("""
Welcome to the Speaker Recognition System! 
This application identifies the speaker from an audio clip. You can either upload a pre-recorded `.wav` file or record your voice directly using your microphone.
""")

st.divider()

# Input options
option = st.radio("Choose input method:", ["Upload Audio File", "Record Audio"])

audio_data = None

if option == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav", "mp3", "ogg"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        audio_data = uploaded_file

elif option == "Record Audio":
    st.info("Click the microphone button to start recording (say something for 3-5 seconds).")
    recorded_audio = st.audio_input("Record your voice")
    if recorded_audio is not None:
        audio_data = recorded_audio

# --- PREDICTION LOGIC ---
if audio_data is not None:
    if st.button("Predict Speaker", type="primary"):
        with st.spinner("Analyzing audio..."):
            try:
                # Extract features
                features = extract_mfcc_mean(audio_data)
                features = features.reshape(1, -1)
                
                # Predict
                if model and le:
                    pred_label_encoded = model.predict(features)
                    pred_label = le.inverse_transform(pred_label_encoded)[0]
                    
                    # Optionally get probabilities if supported
                    try:
                        probs = model.predict_proba(features)[0]
                        max_prob = max(probs)
                        confidence_str = f" ({max_prob*100:.2f}% confidence)"
                    except Exception:
                        confidence_str = ""
                    
                    st.success(f"### Predicted Speaker: **{pred_label}** {confidence_str}")
                    st.balloons()
            except Exception as e:
                st.error(f"An error occurred during processing: {e}")

st.divider()
st.caption("Built for Speech and Image Analysis Course Project.")
