from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    """Handles TF-IDF vectorization and Label Encoding."""

    def __init__(self, max_features=5000):
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        # Initialize the encoder for our target labels (Negative, Neutral, Positive)
        self.label_encoder = LabelEncoder()

    def fit_transform(self, dataframe, text_col='processed_text', label_col='airline_sentiment'):
        """Fits the vectorizer and encoder, then transforms the data."""
        print(f"Extracting features from {text_col}...")

        # 1. Transform text to numbers (Features)
        X = self.vectorizer.fit_transform(dataframe[text_col]).toarray()

        # 2. Transform labels to numbers (Targets)
        y = self.label_encoder.fit_transform(dataframe[label_col])

        return X, y

    def get_input_dim(self):
        # Tells the ANN how many input neurons it needs
        return len(self.vectorizer.get_feature_names_out())