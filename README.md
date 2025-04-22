# Learnability
A Framework for Generative Analytics Using Audibility Aid
---

# Learnability 📚💡

Welcome to **Learnability** – your dynamic study resource platform designed to enhance your learning experience through innovative technology integration! 🚀

## 🌟 Overview

**Learnability** is an intelligent, web-based platform built with Streamlit that empowers users to quickly understand any topic by generating concise summaries from trusted online sources. The application uses transformer-based NLP models (specifically Facebook's bart-large-cnn) for summarization and Google Text-to-Speech (gTTS) to convert those summaries into audio format, enabling both visual and auditory learning.

## 🎯 Key Features

- **Web Scraping 🕸️**: Input your topic of interest, and our sophisticated web scraping mechanism, powered by Beautiful Soup, extracts relevant information from diverse sources across the web.
- **State-of-the-Art NLP Model** : Uses Facebook’s BART transformer model for high-quality abstractive summarization.
- **Auditory Learning 🎧**: Enjoy the flexibility of learning on-the-go with our text-to-speech feature, which converts summaries into audio content, catering to various learning preferences and accessibility needs.

## 🔧 Technology Stack

- **Streamlit**: Our principal framework for crafting a dynamic and interactive user interface.
- **Beautiful Soup**: Efficiently extracts data from an array of web sources.
- **Transformers (HuggingFace)**: For loading and using the facebook/bart-large-cnn summarization model.
- **gTTS (Google Text-to-Speech)**: Converts generated summaries into audio format.

## 🚀 Getting Started

Follow these steps to get up and running with **Learnability**:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/learnability.git
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   ```bash
   streamlit run learnability.py
   ```

## 📚 How It Works

1. **Input Your Topic**: Enter the topic you want to learn about in the user-friendly interface.
2. **Data Extraction**: Our web scraping mechanism gathers relevant information from various websites.
3. **Content Summarization**: Using BART model, the system distills the gathered information into a concise summary.
4. **Audio Option**: Choose to listen to the summarized content using the text-to-speech feature for a flexible learning experience.

## 💡 Why Learnability?

- **Comprehensive**: Access a vast expanse of digital information efficiently.
- **Insightful**: Get key insights and summaries tailored to your needs.
- **Flexible**: Learn in the way that suits you best – read or listen.

## 🙌 Contributing

We welcome contributions from the community! Feel free to fork the repository, make improvements, and submit pull requests. Let's make learning more accessible and engaging together!


## 📜 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Join us on this journey to revolutionize the learning experience! 🌟🚀📚
