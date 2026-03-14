from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class FeatureExtractor:
    """Converts cleaned text into numerical vectors and encodes target labels.
       (Handles TF-IDF vectorization and Label Encoding.)"""

    def __init__(self, max_features=5000):
        """Configures the TF-IDF vectorizer with a limit on the number of features (vocabulary size) to prevent overfitting."""
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        # Initialize the encoder for our target labels (Negative, Neutral, Positive)
        self.label_encoder = LabelEncoder()

    def fit_transform(self, dataframe, text_col='processed_text', label_col='airline_sentiment'):
        """Fits the TF-IDF model to the training text and converts the text categories (Negative, Neutral, Positive)
           into integers (0, 1, 2)."""
        print(f"Extracting features from {text_col}...")

        # 1. Transform text to numbers (Features)
        X = self.vectorizer.fit_transform(dataframe[text_col]).toarray()

        # 2. Transform labels to numbers (Targets)
        y = self.label_encoder.fit_transform(dataframe[label_col])

        return X, y

    def get_input_dim(self):
        """Returns the size of the vocabulary, which determines the number of input neurons needed for the ANN."""
        return len(self.vectorizer.get_feature_names_out())