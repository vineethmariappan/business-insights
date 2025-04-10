import praw
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy
import gensim
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import LdaMulticore

def main():
    nlp = spacy.load("en_core_web_sm")
    nltk.download("stopwords")
    nltk.download("punkt")
    # The following download might be unnecessary unless you have a specific reason for it:
    # nltk.download('punkt_tab')

    # Reddit API Credentials
    reddit = praw.Reddit(
        client_id="9ETAXWFx9IHiC7AOh8-APg",
        client_secret="TuskU4_ysvTwaJJ3i9K40HLEiykE9A",
        user_agent="python:brand-sentiment-analyzer:v1.0 (by /u/Soul_Pay4951)"
    )

    # Choose a subreddit
    subreddit = reddit.subreddit("technology")

    posts = []
    for post in subreddit.hot(limit=5):
        post.comments.replace_more(limit=0)  # Removes 'MoreComments' objects
        comments = [comment.body for comment in post.comments.list()]  # Extracts only comment bodies
        posts.append((post.title, post.selftext, comments))

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
        text = re.sub(r"\W", " ", text)  # Remove special characters
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stopwords.words("english")]
        return " ".join(tokens)

    # Apply cleaning
    cleaned_posts = [(clean_text(title), clean_text(body)) for title, body, _ in posts]
    print(cleaned_posts)

    def extract_entities(text):
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT"]]

    # Extract brand/product mentions
    brands_products = [extract_entities(title + " " + body) for title, body in cleaned_posts]
    print(brands_products)

    # Define stopwords to remove
    stop_words = set(stopwords.words("english"))

    # Preprocess the cleaned posts for LDA (tokenize and remove stopwords)
    def preprocess_for_lda(text):
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
        return tokens

    # Apply preprocessing to cleaned posts
    tokenized_posts = [preprocess_for_lda(title + " " + body) for title, body in cleaned_posts]

    # Create a dictionary from the processed tokens
    dictionary = corpora.Dictionary(tokenized_posts)

    # Convert tokenized posts into a bag-of-words corpus
    corpus = [dictionary.doc2bow(post) for post in tokenized_posts]

    # Apply LDA model using LdaMulticore
    lda_model = LdaMulticore(
        corpus,
        num_topics=3,  # Adjust the number of topics as needed
        id2word=dictionary,
        passes=10,
        workers=2  # Number of threads for parallel processing
    )

    # Display the topics
    topics = lda_model.print_topics(num_words=5)  # Display the top 5 words for each topic
    for topic in topics:
        print(topic)

if __name__ == '__main__':
    main()
