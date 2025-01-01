import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import nltk
import re
from nltk.corpus import stopwords

# Download NLTK data
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Helper function for cleaning text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters and numbers
    text = text.strip()  # Remove extra whitespace
    text = " ".join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Title
st.title("Sentiment Analysis on Customer Reviews")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file with customer reviews", type=["csv"])

if uploaded_file:
    # Load the dataset
    data = pd.read_csv(uploaded_file)

    if "review_text" not in data.columns or "sentiment" not in data.columns:
        st.error("The uploaded file must contain 'review_text' and 'sentiment' columns.")
    else:
        # Display dataset preview
        st.subheader("Dataset Preview")
        st.write(data.head())

        # Data cleaning and preprocessing
        st.subheader("Data Cleaning and Preprocessing")
        data["cleaned_review"] = data["review_text"].apply(clean_text)
        st.write(data[["review_text", "cleaned_review"]].head())

        # Splitting the data
        st.subheader("Train a Sentiment Analysis Model")
        X = data["cleaned_review"]
        y = data["sentiment"]

        # Encode target variable
        sentiment_mapping = {"positive": 2, "neutral": 1, "negative": 0}
        y = y.map(sentiment_mapping)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(max_features=5000)
        X = vectorizer.fit_transform(X).toarray()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Model evaluation
        y_pred = model.predict(X_test)
        st.write("Model Accuracy:", round(model.score(X_test, y_test) * 100, 2), "%")

        # Display classification report
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred, target_names=["negative", "neutral", "positive"]))

        # Upload new reviews for prediction
        st.subheader("Upload New Reviews for Sentiment Prediction")
        new_file = st.file_uploader("Upload a CSV file with 'review_text' column for prediction", type=["csv"])

        if new_file:
            new_data = pd.read_csv(new_file)
            if "review_text" not in new_data.columns:
                st.error("The uploaded file must contain a 'review_text' column.")
            else:
                # Clean and predict sentiments for new reviews
                new_data["cleaned_review"] = new_data["review_text"].apply(clean_text)
                new_X = vectorizer.transform(new_data["cleaned_review"]).toarray()
                new_data["predicted_sentiment"] = model.predict(new_X)
                reverse_mapping = {0: "negative", 1: "neutral", 2: "positive"}
                new_data["predicted_sentiment"] = new_data["predicted_sentiment"].map(reverse_mapping)

                # Display predictions
                st.subheader("Predicted Sentiments")
                st.write(new_data[["review_text", "predicted_sentiment"]])

                # Download the updated dataset
                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode("utf-8")

                csv = convert_df_to_csv(new_data)
                st.download_button(
                    label="Download Predictions",
                    data=csv,
                    file_name="predicted_sentiments.csv",
                    mime="text/csv"
                )
