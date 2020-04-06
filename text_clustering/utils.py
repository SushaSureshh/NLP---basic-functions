''' This folder has utils for different clustering techniques
    Choose the algorithm method suited for you
'''

# Import all libraries

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)


def silhoutte_score(data, min_value, random_state):
    ''' The silhouette_score gives the average value for all the samples.
    This gives a perspective into the density and separation of the formed clusters '''

    for num_clust in range(2, 50):
        clusterer = KMeans(n_clusters=num_clust, random_state=random_state)
        cluster_labels = clusterer.fit_predict(data)
        silhoutte_avg = silhouette_score(data, cluster_labels)
        print("For n_clusters =", num_clust,
              "The average silhouette_score is :", silhoutte_avg)
    if silhoutte_avg >= min_value:
        n_cluster = num_clust
        min_value = silhoutte_avg

    return n_cluster


def k_means_clustering(data, random_state):
    ''' 
    This function is to perform KMeans clustering on the documents.
    '''
    n_cluster = silhoutte_score(data, -50, 10)
    clusterer = KMeans(n_clusters=n_cluster, random_state=random_state)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels


def agglomerative_clustering(data, n_clusters, linkage):
    '''
    Agglomerative Clustering Recursively merges the pair of clusters
    that minimally increases a given linkage distance.
    '''
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels


def db_scan_clustering(data, eps, min_samples):
    '''
    Density-Based Spatial Clustering of Applications with Noise. 
    Finds core samples of high density and expands clusters from them. 
    Good for data which contains clusters of similar density.
    '''
    clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels


def minibatch_k_means_clustering(data, random_state, batch_size):
    ''' 
    This function is to perform mini batch KMeans clustering on the documents. 
    This clustering technique is not as scalable as the k means clustering.
    '''
    n_cluster = silhoutte_score(data, -50, 10)
    clusterer = MiniBatchKMeans(
        n_clusters=n_cluster, random_state=random_state, batch_size=batch_size)
    cluster_labels = clusterer.fit_predict(data)
    return cluster_labels


def show_wordcloud(data, title, background_color, max_words, max_font_size, scale, random_state):
    ''' This function will allow you to visualize your word clusters'''

    wordcloud = WordCloud(
        background_color=background_color,
        stopwords=stopwords,
        max_words=max_words,
        max_font_size=max_font_size,
        scale=scale,
        random_state=random_state  # chosen at random by flipping a coin; it was heads lol
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
