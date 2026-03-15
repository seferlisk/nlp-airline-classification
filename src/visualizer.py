import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

class Visualizer:
    """Handles Exploratory Data Analysis (EDA) and text visualizations."""

    @staticmethod
    def plot_sentiment_distribution(df):
        """Visualizes the class imbalance."""
        plt.figure(figsize=(8, 5))
        sns.countplot(x='airline_sentiment', data=df, hue = 'airline_sentiment',
                      palette='magma', order=['negative', 'neutral', 'positive'])
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

    @staticmethod
    def generate_wordcloud(df, sentiment='negative', text_col='processed_text'):
        """Generates a WordCloud for a specific sentiment."""
        # 1. Filter the data
        sentiment_df = df[df['airline_sentiment'] == sentiment]

        # 2. Combine all tweets into one big string
        all_text = " ".join(sentiment_df[text_col])

        # 3. Create the WordCloud object
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='Reds' if sentiment == 'negative' else 'Greens',
            max_words=100
        ).generate(all_text)

        # 4. Display
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Most Frequent Words in {sentiment.capitalize()} Tweets')
        plt.show()