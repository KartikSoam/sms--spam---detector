import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


st.title("📱 SMS Spam Detector")

st.write("Enter an SMS message to check whether it is Spam or Ham.")


# Load dataset
df = pd.read_csv("sms_spam_100000-1.csv")

df = df.dropna(subset=["message", "label"])


# Input and output
X = df["message"]
y = df["label"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Convert text into numbers
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# Calculate accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)


# Show accuracy
st.subheader("📊 Model Accuracy")
st.write(f"{accuracy * 100:.2f}%")


# SMS input
message = st.text_area("Enter your SMS:")


# Check SMS
if st.button("Check SMS"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")

    else:
        message_tfidf = vectorizer.transform([message])

        prediction = model.predict(message_tfidf)[0]

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ HAM MESSAGE (Not Spam)")  