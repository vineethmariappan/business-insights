import praw
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from transformers import pipeline
import numpy as np
from minisom import MiniSom  # pip install minisom

def initialize_nlp():
    nlp = spacy.load("en_core_web_sm")
    nltk.download("stopwords")
    nltk.download("punkt")
    return nlp

def get_reddit_posts():
    reddit = praw.Reddit(
        client_id="9ETAXWFx9IHiC7AOh8-APg",
        client_secret="TuskU4_ysvTwaJJ3i9K40HLEiykE9A",
        user_agent="python:brand-sentiment-analyzer:v1.0 (by /u/Soul_Pay4951)"
    )
    subreddit = reddit.subreddit("technology")
    posts = []
    for post in subreddit.hot(limit=1000):
        post.comments.replace_more(limit=0)  # Removes 'MoreComments'
        comments = [comment.body for comment in post.comments.list()]
        if len(comments) >= 5:
            posts.append((post.title, post.selftext, comments))
    return posts

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"\W", " ", text)  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

def preprocess_for_lda(text, stop_words):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return tokens

def extract_entities(text, nlp):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

def perform_topic_modeling(tokenized_posts, num_topics=3):
    dictionary = corpora.Dictionary(tokenized_posts)
    corpus = [dictionary.doc2bow(post) for post in tokenized_posts]
    lda_model = LdaMulticore(
        corpus,
        num_topics=num_topics,
        id2word=dictionary,
        passes=10,
        workers=2
    )
    topics = lda_model.print_topics(num_words=5)
    return topics, lda_model, corpus, dictionary

def analyze_sentiments(texts):
    # Using a pretrained sentiment-analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiments = []
    for text in texts:
        result = sentiment_pipeline(text)[0]  # Example: {'label': 'POSITIVE', 'score': 0.99}
        sentiments.append(result)
    return sentiments

# Simplified Hebbian Learning simulation: updating an association score for each topic.
class HebbianLearning:
    def __init__(self, topics):
        self.topics = topics
        self.associations = {topic: 0.0 for topic in topics}

    def update(self, topic, sentiment_score):
        learning_rate = 0.1
        if topic in self.associations:
            self.associations[topic] += learning_rate * sentiment_score

    def get_associations(self):
        return self.associations

# Self-Organizing Maps (SOM) for clustering sentiment trends.
def cluster_sentiments(senti_vectors):
    if not senti_vectors:
        return None
    data = np.array(senti_vectors)
    # Initialize a SOM with a 3x3 grid.
    som = MiniSom(3, 3, data.shape[1], sigma=0.5, learning_rate=0.5, random_seed=42)
    som.random_weights_init(data)
    som.train_random(data, 100)
    # For each sentiment vector, determine its winning node (cluster).
    clusters = [som.winner(vec) for vec in data]
    return clusters

def main():
    # Initialization and Reddit data extraction
    nlp = initialize_nlp()
    posts = get_reddit_posts()

    # Clean posts (title and selftext)
    cleaned_posts = [(clean_text(title), clean_text(body)) for title, body, _ in posts]
    print("Cleaned Posts:")
    for cp in cleaned_posts:
        print(cp)

    # Extract entities (e.g., brands, products) using NER
    brands_products = [extract_entities(title + " " + body, nlp) for title, body in cleaned_posts]
    print("\nExtracted Entities (Brands/Products):")
    for bp in brands_products:
        print(bp)

    # Preprocess text for LDA topic modeling
    stop_words = set(stopwords.words("english"))
    tokenized_posts = [preprocess_for_lda(title + " " + body, stop_words) for title, body in cleaned_posts]
    topics, lda_model, corpus, dictionary = perform_topic_modeling(tokenized_posts, num_topics=3)
    print("\nTopics from LDA:")
    for topic in topics:
        print(topic)

    # Sentiment analysis on combined title and body text for each post
    combined_texts = [title + " " + body for title, body in cleaned_posts]
    sentiments = analyze_sentiments(combined_texts)
    print("\nSentiments:")
    for sentiment in sentiments:
        print(sentiment)

    # Hebbian learning: simulate associations between topics and sentiment scores.
    # Here, topics are represented by their string descriptions.
    topic_strs = [t[1] for t in topics]  # Each topic is a tuple (topic_id, topic_string)
    hebbian = HebbianLearning(topic_strs)
    for res in sentiments:
        # Use positive score for POSITIVE sentiment, negative for others.
        score = res['score'] if res['label'] == "POSITIVE" else -res['score']
        for topic in topic_strs:
            hebbian.update(topic, score)
    associations = hebbian.get_associations()
    print("\nHebbian Associations:")
    for topic, assoc in associations.items():
        print(f"{topic}: {assoc}")

    # For SOM clustering, simulate sentiment vectors.
    # Here, we create a vector [positive_score, negative_score] for each sentiment.
    senti_vectors = []
    for res in sentiments:
        if res['label'] == "POSITIVE":
            senti_vectors.append([res['score'], 0.0])
        else:
            senti_vectors.append([0.0, res['score']])
    clusters = cluster_sentiments(senti_vectors)
    print("\nSOM Clusters:")
    print(clusters)

if __name__ == '__main__':
    main()
