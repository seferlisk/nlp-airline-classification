from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import joblib

class FeatureExtractor:
    """Handles TF-IDF vectorization and Label Encoding."""

    def __init__(self, max_features=5000):
        self.tfidf = TfidfVectorizer(max_features=max_features)
        self.label_encoder = LabelEncoder()

    def fit_transform(self, text_list, labels):
        """Fits the vectorizer and encoder, then transforms the data."""
        X = self.tfidf.fit_transform(text_list).toarray()
        y = self.label_encoder.fit_transform(labels)
        return X, y

    def transform(self, text_list):
        """Transforms new text using the already fitted vectorizer."""
        return self.tfidf.transform(text_list).toarray()

    def get_input_dim(self):
        return len(self.tfidf.get_feature_names_out())