import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    """Handles data visualization for EDA and model performance."""

    @staticmethod
    def plot_sentiment_distribution(dataframe):
        plt.figure(figsize=(8, 5))
        sns.countplot(x='airline_sentiment', data=dataframe, palette='viridis',
                      order=['negative', 'neutral', 'positive'])
        plt.title('Distribution of Airline Sentiments')
        plt.xlabel('Sentiment Class')
        plt.ylabel('Number of Tweets')
        plt.show()

        # Print percentages
        counts = dataframe['airline_sentiment'].value_counts(normalize=True) * 100
        print("Class Percentages:\n", counts)

    @staticmethod
    def plot_tweet_lengths(dataframe):
        dataframe['tweet_length'] = dataframe['processed_text'].apply(lambda x: len(x.split()))
        plt.figure(figsize=(8, 5))
        sns.histplot(dataframe['tweet_length'], bins=20, kde=True, color='skyblue')
        plt.title('Distribution of Word Counts per Tweet')
        plt.show()