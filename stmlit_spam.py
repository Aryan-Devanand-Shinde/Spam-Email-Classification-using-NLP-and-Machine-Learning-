import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model
with open('spam_pickle', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer_pickle', 'rb') as vectorizer_file:
    cv = pickle.load(vectorizer_file)

# Spam prediction function
def predict_spam(text):
    text_vectorized = cv.transform([text])  # Use the loaded vectorizer
    prediction = model.predict(text_vectorized)
    return "Spam" if prediction == 1 else "Not Spam"

# Streamlit app
st.title("Spam Detection System")
st.write("This app detects whether a given SMS or email text is spam or not.")

# Input text from user
user_input = st.text_area("Enter your message:")

# Button to trigger prediction
if st.button("Check Message"):
    if user_input.strip() == "":
        st.error("Please enter a message!")
    else:
        result = predict_spam(user_input)
        st.success(f"The message is classified as: **{result}**")

