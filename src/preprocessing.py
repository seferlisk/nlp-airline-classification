import re

import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary tools for the class
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextPreprocessor:
    """Handles cleaning, tokenization, and normalization of tweets."""

    def __init__(self):
        """Initializes the NLTK Lemmatizer and defines a custom stopword list that preserves negation words
        (no, not, never) to maintain sentiment integrity."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # We keep 'not' and 'no' as they are critical for sentiment
        self.stop_words = self.stop_words - {'not', 'no', 'never', 'neither', 'nor'}

    def clean_text(self, text):
        """Uses Regular Expressions (Regex) to strip @user, URLs, hashtags, punctuation, and digits."""
        # Lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove @user handles
        text = re.sub(r'\@\w+', '', text)
        # Remove hashtags (keeps the word)
        text = re.sub(r'#', '', text)
        # Remove punctuation and digits
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def tokenize_and_lemmatize(self, text):
        """Tokenizes, removes stopwords, and lemmatizes to reduce word variations."""
        tokens = text.split()
        processed = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word not in self.stop_words
        ]
        return " ".join(processed)

    def run(self, dataframe, text_column='text'):
        """Applies the pipeline to the entire dataframe and removes any empty rows generated during processing."""
        print("Preprocessing tweets... please wait.")
        dataframe['cleaned_text'] = dataframe[text_column].apply(self.clean_text)
        dataframe['processed_text'] = dataframe['cleaned_text'].apply(self.tokenize_and_lemmatize)
        # Remove any rows that became empty after cleaning
        dataframe = dataframe[dataframe['processed_text'].str.strip() != ""]
        return dataframe