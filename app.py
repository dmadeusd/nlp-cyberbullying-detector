import streamlit as st
import pickle
import string
import re
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

st.set_page_config(page_title="Cyberbullying Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

download_nltk_data()

en_stopwords = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
punctuation = string.punctuation

def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_tag(tag):
    if tag.startswith("JJ"): return "a"
    elif tag.startswith("V"): return "v"
    elif tag.startswith("R"): return "r"
    else: return "n"
    
def lemmatizing(tokens):
    lemmatized = []
    tagged = pos_tag(tokens)
    for word, tag in tagged:
        label = get_tag(tag.lower())
        if label:
            result = lemmatizer.lemmatize(word, label)
            lemmatized.append(result)
        else:
            result = lemmatizer.lemmatize(word)
            lemmatized.append(result)
    return lemmatized

def preprocess_sentence(sentence):
    sentence = str(sentence)
    sentence = clean_tweet(sentence)
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in en_stopwords]
    tokens = [token for token in tokens if token not in punctuation]
    tokens = lemmatizing(tokens)
    return " ".join(tokens)

@st.cache_resource
def load_models():
    model_path = './Model/logistic_regression_classifier.pkl'
    vectorizer_path = './Model/tfidf_vectorizer.pkl'
    
    with open(model_path, "rb") as f:
        classifier = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
        
    return classifier, vectorizer

classifier, vectorizer = load_models()

st.title("Cyberbullying Comments Detector")
st.markdown("This application uses the logistic regression algorithm.")

st.markdown("---")

user_input = st.text_area("Enter the comment you want to analyze:", height=150, placeholder="Example: You are so ugly and nobody likes you!")

if st.button("Analyze Comment", use_container_width=True, type="primary"):
    if len(user_input.split()) < 2:
        st.warning("Input is too short! Please enter at least 2 words.")
    else:
        with st.spinner('Analyzing text...'):
            processed_text = preprocess_sentence(user_input)
            text_tfidf = vectorizer.transform([processed_text])

            prediction = classifier.predict(text_tfidf)[0]
            probabilities = classifier.predict_proba(text_tfidf)[0]

            max_prob = max(probabilities) * 100
            
            st.markdown("---")
            st.subheader("Analysis Results")

            if prediction == "not_cyberbullying":
                st.success(f"✅ Type: **{prediction.upper()}**")
            else:
                st.error(f"🚨 Detected: **{prediction.upper()}**")
                
            st.markdown(f"**Confidence Level:** {max_prob:.2f}%")

            st.markdown("### Category Probabilities:")

            classes = classifier.classes_

            for idx, class_name in enumerate(classes):
                col1, col2 = st.columns([1.5, 4])
                prob_score = probabilities[idx]
                
                with col1:
                    display_name = class_name.replace('_', ' ').title()
                    st.write(f"**{display_name}**")
                with col2:
                    st.progress(float(prob_score))
