import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from nltk.stem.snowball import SnowballStemmer
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram





film_plot = pd.read_csv('Film plots/movies.csv')

film_plot['plot'] = film_plot['wiki_plot'].astype(str) + '\n' + film_plot['imdb_plot'].astype(str)


# TOKENIZE AND STEM WORDS

stemmer = SnowballStemmer('english')

def tokenize_stem(text):

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)\
              if word.lower() not in nltk.corpus.stopwords.words('english')]
    
    filtered_tokens = [word for word in tokens if re.search('[a-zA-Z]', word)]

    stems = [stemmer.stem(word) for word in filtered_tokens]

    return stems



# Tfidf Vectorizaer

vectorizer = TfidfVectorizer(max_df=0.8, min_df=0.2, max_features=200000, stop_words='english',
                             use_idf=True, tokenizer=tokenize_stem, ngram_range=(1,3))

tfidf_matrix = vectorizer.fit_transform([x for x in film_plot['plot']])




#Kmeans

km = KMeans(n_clusters=5)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

film_plot['Cluster'] = clusters




#Similarity distance

similarity = 1 - cosine_similarity(tfidf_matrix)

mergings = linkage(similarity, method='complete')

dendrogram = dendrogram(mergings, labels = [title for title in film_plot['title']])

plt.show()
