"""
HW3 - Image Clustering
Name - Saksham Nayyar
Username - Sakshamnayyar
G number - G01462522

"""

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

TRAIN_FILE = "test.txt"
OUTPUT_FILE = "format.dat"

def init_centroids_plus_plus(data, k):
    centroids = [random.choice(data)]

    while len(centroids) < k:
        squared_distances = np.array([min([np.linalg.norm(point - centroid) ** 2 for centroid in centroids]) for point in data])
        
        probabilities = squared_distances / sum(squared_distances)
        next_centroid = random.choices(data, probabilities, k=1)
        centroids.append(next_centroid[0])

    return centroids

def assign_to_clusters(data, centroids):
    labels = [np.argmin([np.linalg.norm(point - centroid) for centroid in centroids]) for point in data]
    return labels

def update_centroids(data, assignments, k):
    centroids = []
    for cluster in range(k):
        cluster_points = data[np.array(assignments) == cluster]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
            centroids.append(new_centroid)
        else:
            centroids.append(centroids[-1])

    return centroids

def has_converged(old_centroids, new_centroids, tolerance):
    centroid_distances = [np.linalg.norm(old - new) for old, new in zip(old_centroids, new_centroids)]
    return max(centroid_distances) < tolerance

def k_means(data, k, max_iterations, tolerance):
    centroids = init_centroids_plus_plus(data, k)
    
    for i in range(max_iterations):
        old_centroids = centroids
        labels = assign_to_clusters(data, centroids)
        centroids = update_centroids(data, labels,k)
        if(has_converged(old_centroids,centroids,tolerance)):
            break
        
    return centroids, labels

# Function for fitting the K-means model
def fit(data, k, max_iterations=100, tolerance=1e-4):
    centroids, labels = k_means(data, k, max_iterations, tolerance)
    return centroids, labels

# Function for predicting clusters
def predict(data, centroids):
    clusters = assign_to_clusters(data, centroids)
    return np.array(clusters)

    
def scatter_plot(final_data,labels,k):
    plt.figure(figsize=(12, 10))
    colors = plt.cm.get_cmap('tab10').colors[:k]

    for i, cluster_label in enumerate(np.unique(labels)):
        cluster_data = final_data[labels == cluster_label]  # X is your data
        color = colors[i % len(colors)]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=color, label=f'Cluster {i}')

    plt.title("Clustered Data")
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('Predicted Cluster')
    #plt.savefig('ImageClusters_HW3_Nayyar.png')
    plt.legend()
    plt.show()
    
def calculate_silhouette(data, k):
    centroids, labels = fit(data, k)
    silhouette = silhouette_score(data, labels)
    return silhouette, labels
    
def remove_constant_columns(data):
    constant_columns = np.all(np.isin(data, [0, 255]), axis=0)
    cleaned_dataset = data[:, ~constant_columns]
    return np.array(cleaned_dataset)

def umap_silhouette_scoring(estimator, X, y=None):
    umap_embedding = estimator.transform(X)
    labels = KMeans(n_clusters=10).fit_predict(umap_embedding)
    return silhouette_score(umap_embedding, labels)


if __name__ == "__main__":
    #loading the data form train file 
    data = np.loadtxt(TRAIN_FILE, delimiter=',', dtype=int)
    
    #Removing the constant columns
    data = remove_constant_columns(data)
    
    #Performing MinMaxScaler to normalize data
    st = MinMaxScaler()
    data = st.fit_transform(data)

    #Used UMAP to reduce dimensionality of the data
    umap_model = UMAP(n_components=2)
    data = umap_model.fit_transform(data)
    
    #Fitting the model with data
    n_clusters = 10
    centroids, labels = fit(data, n_clusters)
    
    #Plotting the labels
    scatter_plot(data, labels, n_clusters)
    
    #Computing silhouette coefficient for the predicted labels
    sc = silhouette_score(data, labels)
    print(sc)
    
    #Loading data to a text file
    with open(OUTPUT_FILE, "w") as file:
        for item in labels:
            file.write(str(item) + "\n")
            
    
    #To calculate silhouette Coefficient value for each value of K increasing from 2 to 20 in steps of 2 for the data.
    """possible_k_values = range(2,21,2)  # Adjust the range based on your requirements
    silhouette_scores = []  
    preds = []
    
    for k_value in possible_k_values:
        sc,labels = calculate_silhouette(data, k_value)
        print(sc)
        silhouette_scores.append(sc)"""
    
    #To find the best parameters of UMAP using RandomizedSearchCV
    """
    param_dist = {
        'n_neighbors': randint(5, 20),
        'min_dist': uniform(0.1, 1.0),
        'n_components': np.arange(2, 701, 10).tolist(),
    }
    umap_model = UMAP()
    randomized_search = RandomizedSearchCV(umap_model, param_distributions=param_dist, n_iter=20, cv=3,scoring=umap_silhouette_scoring, verbose = 2)
    randomized_search.fit(data)
    best_params = randomized_search.best_params_"""
    
    


