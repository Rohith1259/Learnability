import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from gtts import gTTS  
import tempfile
import os

# Load model and tokenizer
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

@st.cache_data(show_spinner=False)
def fetch_wikipedia(query):
    try:
        response = requests.get(f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}")
        soup = BeautifulSoup(response.text, "html.parser")
        content = " ".join([p.text for p in soup.find_all("p")[:10]])
        return content.strip()
    except Exception:
        return ""

@st.cache_data(show_spinner=False)
def fetch_geeksforgeeks(query):
    try:
        url = f"https://www.geeksforgeeks.org/{query.replace(' ', '-')}/"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        content = " ".join([p.text for p in soup.find_all("p")[:10]])
        return content.strip()
    except Exception:
        return ""

def generate_chunks(text, chunk_size=1024):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

def summarize_text(text):
    try:
        summaries = []
        for chunk in generate_chunks(text):
            inputs = tokenizer.encode(chunk, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(
                inputs,
                max_length=500,
                min_length=200,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary)
        return "\n\n".join(summaries)
    except Exception as e:
        return f"Summarization error: {str(e)}"

# Updated text-to-speech using gTTS
def text_to_speech(summary_text):
    try:
        tts = gTTS(text=summary_text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
            tts.save(tmpfile.name)
            return tmpfile.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Learnability", layout="centered")
st.title("\U0001F9E0 Learnability - Smart Topic Summarizer")

# Sidebar
with st.sidebar:
    st.markdown('<div class="side-panel">', unsafe_allow_html=True)
    st.header("Learnability Info and Team")

    with st.expander("About"):
        st.write("Welcome to Learnability !\n\n A cutting-edge platform empowering learners with rapid text summarization from different sources, optimizing time utilization. "
                 "Elevate your learning journey through the sophisticated text-to-speech feature. Embrace a seamless path to acquiring knowledge effortlessly, fostering remarkable growth and proficiency.")

    with st.expander("Profiles"):
        st.markdown("#### - Katuri Sreeja")
        st.write("Hey there, I'm Sreeja.\n\nAn AI and Data Science undergraduate with a passion for exploring AI, data science, and machine learning. I intend to enhance my skill set by delving into these technologies and generate significant solutions.")

        st.markdown("#### - M . Rohith")
        st.write("Hello, I'm Rohith.\n\nAs an Artificial Intelligence and Data Science I love to explore advanced technologies in Artificial Intelligence , Data Science and create meaningful solutions.")

        st.markdown("#### - D . Siddharth Ram ")
        st.write("Hi, I'm Siddharth.\n\nAn Artificial Intelligence and Data Sceince undergrad with an interest in cutting-edge technologie\u00ads such as artificial intelligence, data scie\u00adnce, and machine learning. I aim to attain expertise\u00ad in web developme\u00adnt someday!")

    with st.expander("Feedback and Contact"):
        st.write("We would love to hear your feedback about Learnability. If you have any questions, suggestions, "
                 "or issues, please feel free to reach out to us using the contact information below:")

        st.markdown("\u2709 Email: learnability03@gmail.com ")

# Main input
topic = st.text_input("Enter a topic you want to learn about:")

if st.button("Generate Summary") and topic:
    with st.spinner("Fetching and summarizing data..."):
        wiki_content = fetch_wikipedia(topic)
        gfg_content = fetch_geeksforgeeks(topic)

        if not wiki_content and not gfg_content:
            st.error("No content found from either source.")
        else:
            combined = wiki_content + "\n" + gfg_content
            summary = summarize_text(combined)
            st.subheader("\U0001F4C4 Generated Summary")
            st.write(summary)

            # Audio
            st.subheader("🔊 Listen to Summary")
            audio_file_path = text_to_speech(summary)
            if audio_file_path:
                audio_bytes = open(audio_file_path, 'rb').read()
                st.audio(audio_bytes, format='audio/mp3')
                os.remove(audio_file_path)  # Clean up temp file
