import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:
    """Handles Exploratory Data Analysis (EDA) visualizations."""

    @staticmethod
    def plot_sentiment_distribution(df):
        """Visualizes the class imbalance."""
        plt.figure(figsize=(8, 5))
        sns.countplot(x='airline_sentiment', data=df, palette='viridis',
                      order=['negative', 'neutral', 'positive'])
        plt.title('Airline Sentiment Distribution (Raw Data)')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Tweets')
        plt.show()

    @staticmethod
    def plot_airline_distribution(df):
        """Shows which airlines have the most tweets."""
        plt.figure(figsize=(10, 6))
        sns.countplot(x='airline', data=df, hue='airline_sentiment', palette='magma')
        plt.title('Sentiment Distribution per Airline')
        plt.xticks(rotation=45)
        plt.show()

    @staticmethod
    def plot_tweet_length(df, text_col='text'):
        """Analyzes the length of tweets (number of words)."""
        lengths = df[text_col].apply(lambda x: len(x.split()))
        plt.figure(figsize=(8, 5))
        sns.histplot(lengths, bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Tweet Word Counts')
        plt.xlabel('Number of Words')
        plt.show()