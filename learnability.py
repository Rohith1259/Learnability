import streamlit as st
import urllib.parse
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pyttsx3
import os
import time  # Import the time module for the delay

# Function to scrape relevant data from URLs
def scrape_data_from_urls(urls, keyword):
    texts = []
    with requests.Session() as session:
        for url in urls:
            try:
                url_with_keyword = urllib.parse.urljoin(url, keyword)
                response = session.get(url_with_keyword)
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.content, "html.parser")

                # Specify the relevant tag or tags for scraping the required information
                relevant_elements = soup.find_all('p') + soup.find_all('div', class_='relevant-class')
                relevant_text = ' '.join([element.get_text(strip=True) for element in relevant_elements])
                texts.append(relevant_text)

            except requests.RequestException as e:
                print(f"Error occurred while generating")
                continue  # Skip this website and continue with the next one

    return texts

# Function to generate a summary using TF-IDF
def generate_summary(text, num_sentences=100):
    sentences = sent_tokenize(text)
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    sentence_matrix = vectorizer.fit_transform(sentences)
    cosine_sim_matrix = cosine_similarity(sentence_matrix, sentence_matrix)
    sentence_scores = {i: sum(cosine_sim_matrix[i]) - cosine_sim_matrix[i][i] for i in range(len(sentences))}
    top_sentences_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = [sentences[i] for i in top_sentences_indices]
    return "\n".join(summary)



def text_to_speech(text):
    engine = pyttsx3.init()
    engine.save_to_file(text, 'output.mp3')
    engine.runAndWait()
    return 'output.mp3'

def main():
    st.title("Learnability")

    # Custom CSS styling for the layout
    st.markdown(
        """
        <style>
        .main {
            padding: 0;
            background-image: linear-gradient(to right, #1167B1 ,#77CFF2); /* Background gradient from right to left */
            border-radius: 10px;
            color: #ffffff; /* White text color */
            font-family: 'Helvetica', sans-serif; /* Font family for the title */
        }
        .input-area {
            margin-bottom: 1rem;
        }
        .summary-area {
            margin-top: 1rem;
            width: 100%;
            height: 10rem;
            border: 1px solid #ccc;
            border-radius: 5px;
            padding: 1rem;
            background-color: #ffffff;
        }
        .btn-generate {
            margin-right: 1rem;
        }
        .btn-audio {
            background-color: green;
            color: #ffffff;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .side-panel {
            padding: 0;
            background-color: #333333;
            border-radius: 10px;
            text-align: justify;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main content area
    st.markdown('<div class="main">', unsafe_allow_html=True)
     
     
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="side-panel">', unsafe_allow_html=True)
        st.header("Learnability Info and Team")

        # "About" section (Dropdown)
        with st.expander("About"):
            st.write("Welcome to Learnability !\n\n A cutting-edge platform empowering learners with rapid text summarization from different sources, optimizing time utilization. "
                 "Elevate your learning journey through the sophisticated text-to-speech feature. Embrace a seamless path to acquiring knowledge effortlessly, fostering remarkable growth and proficiency.")

        # Profiles section (Dropdown)
        with st.expander("Profiles"):
            st.markdown("#### - Katuri Sreeja")
            st.write("Hey there, I'm Sreeja.\n\nAn AI and Data Science undergraduate with a passion for exploring AI, data science, and machine learning. I intend to enhance my skill set by delving into these technologies and generate significant solutions.")

            st.markdown("#### - M . Rohith")
            st.write("Hello, I'm Rohith.\n\nAs an Artificial Intelligence and Data Science I love to explore advanced technologies in Artificial Intelligence , Data Science and create meaningful solutions.")
          

            st.markdown("#### - D . Siddharth Ram ")
            st.write("Hi, I'm Siddharth.\n\nAn Artificial Intelligence and Data Sceince undergrad with an interest in cutting-edge technologie­s such as artificial intelligence, data scie­nce, and machine learning. I aim to attain expertise­ in web developme­nt someday!")

        # Feedback and Contact section (Dropdown)
        with st.expander("Feedback and Contact"):
            st.write("We would love to hear your feedback about Learnability. If you have any questions, suggestions, "
                     "or issues, please feel free to reach out to us using the contact information below:")

            st.markdown("✉ Email: learnability03@gmail.com ")


        st.markdown('</div>', unsafe_allow_html=True)

    # Text input for user input
    with st.container():
        user_input = st.text_input("Enter your keywords:", key="user_input")
        st.markdown('<style>.css-16p5q05 textarea { margin-bottom: 1rem; }</style>', unsafe_allow_html=True)

    # Buttons area
    with st.container():
        if st.button("Generate Summary", key="btn_generate"):
            if not user_input.strip():  # Check if the input is empty or contains only whitespace
                st.warning("Please enter a keyword before generating the summary.")
            else:
                # List of URLs to scrape data from
                urls = ["https://www.google.co.in/", "https://www.geeksforgeeks.org/", "https://en.wikipedia.org/wiki/Wikipedia:"]
                
                
                # Scrape the data from the URLs for the given keyword
                data = scrape_data_from_urls(urls, user_input)
                
                # Join the list of texts into one string
                data = " ".join(data)
                
                # Show the spinner while processing the summary generation
                with st.spinner("Generating summary..."):
                    # Simulate processing time (5 seconds delay in this example)
                    time.sleep(1)
                    
                    # Generate the summary using TF-IDF
                    summary = generate_summary(data, num_sentences=3)
                    
                    # Display the summary
                    st.text_area("Summary:", summary, key="summary_area", height=10)
                    st.markdown('<style>.css-1e7m3am { margin-top: 1rem; border: 1px solid #ccc; border-radius: 5px; padding: 1rem; background-color: #ffffff; }</style>', unsafe_allow_html=True)

        if st.button("Text-to-Speech", key="btn_audio"):
            if user_input:
                # List of URLs to scrape data from
                urls = ["https://www.google.co.in/", "https://www.geeksforgeeks.org/", "https://en.wikipedia.org/wiki/Wikipedia:"]
                
                # Scrape the data from the URLs for the given keyword
                data = scrape_data_from_urls(urls, user_input)
                
                # Join the list of texts into one string
                data = " ".join(data)
                
                # Generate the summary using TF-IDF
                summary = generate_summary(data, num_sentences=3)
                
                # Convert the summary to speech
                audio_file = text_to_speech(summary)
                st.audio(audio_file, format='audio/mp3')

            else:
                st.warning("Please enter a keyword before generating speech.")
   

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()