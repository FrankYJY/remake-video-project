from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def cluster(points,eps=0.3):
    '''
    features, true_labels = make_moons(
        n_samples=250, noise=0.05, random_state=42
    )
    '''
    features = points
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Instantiate k-means and dbscan algorithms
    #kmeans = KMeans(n_clusters=2)
    dbscan = DBSCAN(eps=eps)

    # Fit the algorithms to the features
    #kmeans.fit(scaled_features)
    dbscan.fit(scaled_features)

    # Compute the silhouette scores for each algorithm
    '''
    kmeans_silhouette = silhouette_score(
        scaled_features, kmeans.labels_
    ).round(2)
    '''
    dbscan_silhouette = silhouette_score(
    scaled_features, dbscan.labels_
    ).round (2)

    print(features)
    print(dbscan.labels_)
    return dbscan.labels_
    '''
    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 6), sharex=True, sharey=True
    )
    fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
    }
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
    ax1.set_title(
        f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
    )

    # The dbscan plot
    db_colors = [fte_colors[label] for label in dbscan.labels_]
    ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
    ax2.set_title(
        f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
    )
    plt.show()
    input()
    '''