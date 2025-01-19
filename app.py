# app.py
import streamlit as st
import os
import numpy as np
import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    pipeline,
    AutoTokenizer, 
    AutoModelForSeq2SeqGeneration
)
from groq import Groq
from groq.types import ChatCompletion
import time
import io
from googletrans import Translator
import librosa
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
try:
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
except Exception as e:
    st.error(f"Error initializing Groq client: {str(e)}")

# Initialize models and processors
@st.cache_resource
def load_speech_models():
    try:
        processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        return processor, model
    except Exception as e:
        st.error(f"Error loading speech models: {str(e)}")
        return None, None

@st.cache_resource
def load_translation_model():
    try:
        model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading translation model: {str(e)}")
        return None, None

# Load models
whisper_processor, whisper_model = load_speech_models()
translation_tokenizer, translation_model = load_translation_model()

# Page configuration
st.set_page_config(
    page_title="Language Learning Buddy",
    page_icon="🌎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .bot-message {
        background-color: #F5F5F5;
    }
    .phrase-card {
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2C3E50;
    }
    .stProgress > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Utility Functions
def process_audio(audio_data, sample_rate=16000):
    """Process audio data for the Whisper model"""
    try:
        # Convert to float32 and normalize
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        # Resample if necessary
        if sample_rate != 16000:
            audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        return audio_data
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

def speech_to_text(audio_data, language='en'):
    """Convert speech to text using Whisper"""
    try:
        if whisper_processor is None or whisper_model is None:
            raise ValueError("Speech models not properly initialized")
            
        # Process audio data
        processed_audio = process_audio(audio_data)
        if processed_audio is None:
            return None
            
        # Convert to tensor
        input_features = whisper_processor(
            processed_audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        # Generate token ids
        predicted_ids = whisper_model.generate(input_features)
        
        # Decode token ids to text
        transcription = whisper_processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription.strip()
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return None

def get_ai_response(prompt, language='en'):
    """Get AI response using Groq"""
    try:
        completion: ChatCompletion = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful language learning assistant. Respond in {language} with natural, conversational language. Keep responses concise and focused on helping the user learn the language."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting AI response: {str(e)}")
        return "I apologize, but I'm having trouble generating a response at the moment."

def translate_text(text, source_lang, target_lang):
    """Translate text using translation service"""
    try:
        translator = Translator()
        translation = translator.translate(
            text,
            src=source_lang[:2].lower(),
            dest=target_lang[:2].lower()
        )
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return None

def record_audio(duration=5, sample_rate=16000):
    """Record audio from microphone"""
    try:
        st.info("🎤 Recording...")
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        return recording
    except Exception as e:
        st.error(f"Error recording audio: {str(e)}")
        return None

def analyze_pronunciation(recorded_text, expected_text):
    """Analyze pronunciation accuracy"""
    try:
        recorded_words = set(recorded_text.lower().split())
        expected_words = set(expected_text.lower().split())
        
        correct_words = len(recorded_words.intersection(expected_words))
        total_words = len(expected_words)
        accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
        
        if accuracy >= 90:
            feedback = "Excellent pronunciation! Keep it up! 🌟"
        elif accuracy >= 70:
            feedback = "Good job! Some room for improvement. 👍"
        else:
            feedback = "Keep practicing! Try to focus on word clarity. 💪"
        
        return {
            "accuracy": accuracy,
            "feedback": feedback,
            "matched_words": list(recorded_words.intersection(expected_words)),
            "missed_words": list(expected_words - recorded_words)
        }
    except Exception as e:
        st.error(f"Error analyzing pronunciation: {str(e)}")
        return None

def get_phrases(category, source_lang, target_lang):
    """Get phrases for selected category with translations"""
    base_phrases = {
        "Greetings": [
            "Hello",
            "Good morning",
            "How are you?",
            "Nice to meet you",
            "Goodbye"
        ],
        "Travel": [
            "Where is the airport?",
            "How much does it cost?",
            "I need a taxi",
            "Can you help me?",
            "Which way to the hotel?"
        ],
        "Emergency": [
            "Help!",
            "I need a doctor",
            "Call the police",
            "Where is the hospital?",
            "It's an emergency"
        ]
    }
    
    try:
        phrases = base_phrases.get(category, [])
        translated_phrases = []
        
        for phrase in phrases:
            translation = translate_text(phrase, "en", target_lang)
            translated_phrases.append({
                "source": phrase,
                "target": translation,
                "context": category
            })
        
        return translated_phrases
    except Exception as e:
        st.error(f"Error getting phrases: {str(e)}")
        return []

def main():
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    st.sidebar.title("🌍 Language Settings")
    source_lang = st.sidebar.selectbox(
        "Your Language",
        ["English", "Spanish", "French", "German", "Italian"]
    )
    target_lang = st.sidebar.selectbox(
        "Learning Language",
        ["Spanish", "French", "German", "Italian", "English"],
        index=0 if source_lang != "Spanish" else 1
    )
    
    # Main title
    st.title("🎓 Language Learning Buddy")
    st.markdown("### Your AI-powered language learning assistant")
    
    # Main tabs
    tabs = st.tabs(["💭 Chat", "🎙️ Pronunciation", "📚 Phrases", "🔄 Translation"])
    
    # Chat Tab
    with tabs[0]:
        st.header("Interactive Chat Practice")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        user_input = st.chat_input("Type your message...")
        if user_input:
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            with st.spinner("Getting response..."):
                ai_response = get_ai_response(user_input, target_lang)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": ai_response
                })
            
            st.rerun()
    
    # Pronunciation Tab
    with tabs[1]:
        st.header("Pronunciation Practice")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_to_pronounce = st.text_input(
                "Enter text to practice:",
                "Hello, how are you?"
            )
        
        with col2:
            st.write("Record your pronunciation:")
            if st.button("🎤 Start Recording"):
                with st.spinner("Recording..."):
                    audio_data = record_audio(3)
                    if audio_data is not None:
                        st.audio(audio_data, sample_rate=16000)
                        
                        with st.spinner("Analyzing pronunciation..."):
                            recognized_text = speech_to_text(audio_data)
                            if recognized_text:
                                st.write("Recognized text:", recognized_text)
                                
                                analysis = analyze_pronunciation(
                                    recognized_text,
                                    text_to_pronounce
                                )
                                
                                if analysis:
                                    st.write(f"Accuracy: {analysis['accuracy']:.1f}%")
                                    st.write("Feedback:", analysis['feedback'])
                                    if analysis['missed_words']:
                                        st.write("Words to practice:",
                                               ", ".join(analysis['missed_words']))
    
    # Phrases Tab
    with tabs[2]:
        st.header("Phrase Library")
        
        categories = ["Greetings", "Travel", "Emergency"]
        selected_category = st.selectbox("Select category:", categories)
        
        with st.spinner("Loading phrases..."):
            phrases = get_phrases(selected_category, source_lang, target_lang)
            
            for phrase in phrases:
                with st.expander(f"{phrase['source']} ➜ {phrase['target']}"):
                    st.write("Context:", phrase['context'])
    
    # Translation Tab
    with tabs[3]:
        st.header("Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            text_to_translate = st.text_area("Enter text to translate:")
            if st.button("Translate"):
                if text_to_translate:
                    with st.spinner("Translating..."):
                        translation = translate_text(
                            text_to_translate,
                            source_lang,
                            target_lang
                        )
                        if translation:
                            st.success(translation)
        
        with col2:
            st.write("Voice Translation")
            if st.button("🎤 Record for Translation"):
                with st.spinner("Recording..."):
                    audio_data = record_audio(5)
                    if audio_data is not None:
                        with st.spinner("Processing speech..."):
                            recognized_text = speech_to_text(audio_data)
                            if recognized_text:
                                st.write("Recognized text:", recognized_text)
                                with st.spinner("Translating..."):
                                    translation = translate_text(
                                        recognized_text,
                                        source_lang,
                                        target_lang
                                    )
                                    if translation:
                                        st.success(f"Translation: {translation}")

if __name__ == "__main__":
    main()