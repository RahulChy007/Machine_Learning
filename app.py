import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------------
# Download required NLTK data (runs only if not already downloaded)
# -------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# -------------------------
# Initialize Stemmer & Stopwords
# -------------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------------
# Text Preprocessing Function
# -------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    filtered_words = []

    # Remove special characters
    for word in text:
        if word.isalnum():
            filtered_words.append(word)

    # Remove stopwords
    final_words = []
    for word in filtered_words:
        if word not in stop_words and word not in string.punctuation:
            final_words.append(word)

    # Stemming
    stemmed_words = []
    for word in final_words:
        stemmed_words.append(ps.stem(word))

    return " ".join(stemmed_words)


# -------------------------
# Load Model & Vectorizer
# -------------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


# -------------------------
# Streamlit UI
# -------------------------
st.title("📩 Email / SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    if input_sms.strip() == "":
        st.warning("Please enter a message.")
    else:
        # 1️⃣ Preprocess
        transformed_sms = transform_text(input_sms)

        # 2️⃣ Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3️⃣ Predict
        result = model.predict(vector_input)[0]

        # 4️⃣ Display Result
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
