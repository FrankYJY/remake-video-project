from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import numpy as np

def cluster(points,eps=1,n_clusters=2):
    '''
    features, true_labels = make_moons(
        n_samples=250, noise=0.05, random_state=42
    )
    '''
    selectPoints=[[item[0],item[1],1] for item in points]
    xmean=np.mean([item[0] for item in points])
    ymean = np.mean([item[1] for item in points])
    #print(xmean,ymean)
    '''
    for item in selectPoints:
        if(((item[0]-xmean)*(item[0]-xmean)+(item[1]-ymean)*(item[1]-ymean))>1280):
            item[2]=0
    '''
    selectedPoints=[]
    for item in selectPoints:
        if(item[2]==1):
            selectedPoints.append((item[0],item[1]))
    features = selectedPoints
    scaler = StandardScaler()
    #scaled_features = scaler.fit_transform(features)
    
    scaled_features =features 
    # Instantiate k-means and dbscan algorithms
    kmeans = KMeans(n_clusters=n_clusters, init = 'k-means++')
    dbscan = DBSCAN(eps=eps)

    # Fit the algorithms to the features
    kmeans.fit(scaled_features)
    dbscan.fit(scaled_features)

    # Compute the silhouette scores for each algorithm
    '''
    kmeans_silhouette = silhouette_score(
        scaled_features, kmeans.labels_
    ).round(2)
    '''
    
    
    # dbscan_silhouette = silhouette_score(
    # scaled_features, dbscan.labels_
    # ).round (2)

    # print(features)
    # print(dbscan.labels_)
    
    
    #plt.scatter([item[0] for item in features],[item[1] for item in features],label="stars")
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
        2:"#00d724",
        3:'#000000'
    }
    km_colors = [fte_colors[label] for label in range(n_clusters)]
    plt.figure(figsize=(8,8))
    #print(scaled_features[:][0])
    #print(scaled_features[:][1])
    plt.scatter([item[0] for item in features],[item[1] for item in features], c=dbscan.labels_,cmap="coolwarm")
    
    plt.scatter([xmean],[ymean], s=50,c='b')
    plt.show()
    '''
    # Plot the data and cluster silhouette comparison
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8, 8), sharex=True, sharey=True
    )
    fig.suptitle(f"Clustering Algorithm Comparison: Crescents", fontsize=16)
    fte_colors = {
        0: "#008fd5",
        1: "#fc4f30",
        2:"#00d724",
        3:'#000000'
        
    }
    # The k-means plot
    km_colors = [fte_colors[label] for label in kmeans.labels_]
    ax1.scatter(scaled_features[:][0], scaled_features[:][1], c=km_colors)
    ax1.set_title(
        f"k-means\nSilhouette: {kmeans_silhouette}", fontdict={"fontsize": 12}
    )
    plt.show()
    
    # The dbscan plot
    
    db_colors = [fte_colors[label] for label in dbscan.labels_]
    ax2.scatter(scaled_features[:, 0], scaled_features[:, 1], c=db_colors)
    ax2.set_title(
        f"DBSCAN\nSilhouette: {dbscan_silhouette}", fontdict={"fontsize": 12}
    )
    
    plt.show()
    '''
    slabels=dbscan.labels_
    res =[]
    idx=0
    for item in selectPoints:
        if(item[2]==1):
            res.append(slabels[idx])
            idx+=1
        else:
            res.append(0)
    return res

def get_cluster_labels_descending(cluster_labels):
    # return [[label, count],...] in count descending order
    cluster_labels_counts = Counter(cluster_labels)
    cluster_labels_counts_list = []
    for key in cluster_labels_counts:
        cluster_labels_counts_list.append([key, cluster_labels_counts[key]])
    cluster_labels_counts_list.sort(key = lambda x: x[1], reverse=True)
    # max_cluster_labels_counter_val = 0

    # for key in cluster_labels_counter:
    #     if max_cluster_labels_counter_val < cluster_labels_counter[key]:
    #         max_cluster_labels_counter_val = cluster_labels_counter[key]
    #         bkg_label = key
    # print(cluster_labels_counter, bkg_label)
    return cluster_labels_counts_list