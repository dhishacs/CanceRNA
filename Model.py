#MODEL BUILDING
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import matplotlib.style as style

# USER DEFINED FUNCTIONS
# Function to display a portion of the DataFrame
def display_dataframe(df, num_rows=10):
    st.dataframe(df.head(num_rows))
# Function to display a subset of columns
def display_dataframe_columns(df, num_columns=10):
    st.dataframe(df.iloc[:, :num_columns])
# Function to balance the dataset with smote
def balance_class_smote(X, y):
    """X and Y smote over sampling
    args:
        X feature
        y label
    returns:
        smote X feature
        smote y label
    """
    try:
        smote_over_sampling = SMOTE(random_state=50, n_jobs=-1)    
        X, y = smote_over_sampling.fit_resample(X, y)
    except Exception as e:            
        print(f"Error in SMOTE: {e}")
        # You can log the error if needed
    return X, y
# Function to perform principal component analysis
def pca_x_reduction(pca, X):
    """ X feature reduction
    args:
        pca class object
        X feature
    returns:
        X new feature
        PCA explained variance
        cumulative sum of eigenvalues
    """
    try:
        X = pca.fit_transform(X)
        pca_explained_variance = pca.explained_variance_ratio_                
        cumulative_sum_eigenvalues = np.cumsum(pca_explained_variance)
    except Exception as e:               
        print(f"Error in PCA: {e}")
        # You can log the error if needed
    return X, pca_explained_variance, cumulative_sum_eigenvalues
# Function to find out the optimum number of clusters
def optimize_k_means(data,max_k):
    means=[]
    inertias=[]
    for k in range(1,max_k):
        kmeans=KMeans(n_clusters=k)
        kmeans.fit(data)
        
        means.append(k)
        inertias.append(kmeans.inertia_)
    
    #GENERATING ELBOW PLOT
    fig = plt.subplots(figsize=(10,5))
    plt.plot(means,inertias,"o-")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    st.markdown("### Elbow Method to determine the optimal K value")
    st.pyplot(plt)

# Function to plot 3D scatter plot
def plot_3d_scatter(X, labels):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis', edgecolor='k', s=50)
    
    # Adding labels
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    
    # Adding colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster Label")
    
    plt.title("3D Scatter Plot of Cancer Gene Expression RNA-Seq with PCA Components")
    st.pyplot(plt)

#READING THE INPUT DATA
X = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\data.csv")
# Drop the first column
X = X.drop(X.columns[0], axis=1)

#READING THE CLASSIFICATION LABELS
Y = pd.read_csv("F:\SAMYUGTHA\PSGCT\SEMESTER 5\ML\ML CA 3\labels.csv")
# Drop the first column
Y = Y.drop(Y.columns[0], axis=1)

# PAGE CONFIGURATIONS
st.set_page_config(
    page_title="Model",
    page_icon=":tada:",
    layout="wide",
    initial_sidebar_state="collapsed",  # Collapsed sidebar for wider view
)

# HEADER SECTION
st.title("K- MEANS CLUSTERING MODEL")

st.markdown("### INTRODUCTION")
st.write("Clustering is a set of techniques used to partition data into groups, or clusters. Clusters are loosely defined as groups of data objects that are more similar to other objects in their cluster than they are to data objects in other clusters.The primary goal of clustering is the grouping of data into clusters based on similarity, density, intervals or particular statistical distribution measures of the data space.")
st.write("In this ML paper the K-Means clustering algorithm will be applied to build a predicted model for a gene expression RNA-Seq dataset. This clustering algorithm is an Unsupervised Learning technique used to identify clusters of multidimensional data objects in a dataset. The K-Means clustering algorithm is part of the popular ML scikit-learning framework library.")

# Displaying a count plot for the first column of the dataset Y
st.markdown("### Cancer Gene Expression RNA-Seq Dataset")
plt.figure(figsize=(8, 6))
sns.countplot(x=Y.iloc[:, 0])
plt.xlabel("Imbalanced Cancer Class")
plt.ylabel("Count")
st.pyplot(plt)
st.write(" The bar chart shown above contains the imbalanced class for our selected genomic dataset.")

column_names_list = X.columns.tolist()
# Balancing the unbalanced data using SMOTE
X_balanced, y_balanced = balance_class_smote(X, Y.iloc[:, 0])

#Scaling the Balanced input
X_new=pd.DataFrame()
scaler = StandardScaler()    
X_new[column_names_list] = scaler.fit_transform(X_balanced[column_names_list])

# Displaying a count plot for the Balanced Dataset
st.markdown("### Balanced Cancer Gene Expression RNA-Seq Dataset")
plt.figure(figsize=(8, 6))
sns.countplot(x=y_balanced)
plt.xlabel("Balanced Cancer Class")
plt.ylabel("Count")
st.pyplot(plt)
st.write(" The bar chart shown above contains the imbalanced class for our selected genomic dataset.")

#PRINTING THE NEW DATASET
# Display a portion of the DataFrame
st.markdown("### STANDARDIZED DATA")
st.write("Balancing the unbalanced data using Synthetic Minority Oversampling Technique (SMOTE) and StandardScaler")
st.markdown("### Preview of the Dataset")
display_dataframe(X_new)

# Display a subset of columns
st.markdown("### Subset of Columns")
display_dataframe_columns(X_new)

# Encoding the labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_balanced)

# To get the mapping of encoded labels to original class names
class_names = label_encoder.classes_
print("Encoded Labels:", y_encoded)
print("Original Class Names:", class_names)

#PCA REDUCTION
st.markdown("### PCA REDUCTION")
st.write("If PCA is for data visualization, then we should select 2 or 3 principal components.")
st.write("For a 2D plot, we need to select 2 principal components.")
st.write("For a 3D plot, we need to select 3 principal components.")

st.write("If we want an exact amount of variance to be kept in data after applying PCA,we specify a float between 0 and 1 to the hyperparameter n_components.")
pca = PCA(n_components=0.80)
n_components = pca.n_components
# Initialize PCA with the specified number of components
pca = PCA(n_components=n_components, random_state=50)
X_pca, pca_explained_variance, cumulative_sum_eigenvalues = pca_x_reduction(pca, X_new)

st.markdown("### The Explained Variance Plot")
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(cumulative_sum_eigenvalues) + 1), cumulative_sum_eigenvalues, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Plot')
plt.grid(True)
#Create the bar graph
plt.subplot(1, 2, 2)
plt.bar(range(1, len(pca_explained_variance) + 1), pca_explained_variance, color='g', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Principal Component')
plt.grid(True)
st.pyplot(plt)

st.markdown("### Scree Plot")
plt.figure(figsize=(8, 6))
plt.subplot(1, 1, 1)
plt.style.use("ggplot") 
plt.plot(pca.explained_variance_, marker='o')
plt.xlabel("Eigenvalue number")
plt.ylabel("Eigenvalue size")
plt.title("Scree Plot")
st.pyplot(plt)

# Print the explained variance and cumulative sum of eigenvalues
st.write("Explained Variance:", pca_explained_variance)
st.write("Cumulative Sum of Eigenvalues:", cumulative_sum_eigenvalues)

# Plotting the Cancer Gene Expression RNA-Seq with Principal Component 1 and Principal Component 2
st.markdown("### Scatter Plot of Cancer Gene Expression RNA-Seq with PCA Components")
plt.figure(figsize=(10, 8))
#plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_encoded, cmap='binary', edgecolor='k', s=50)
plt.scatter(X_pca[:, 0], X_pca[:, 1], edgecolor='k', s=50)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cancer Class")
st.pyplot(plt)

#IDENTIFYING THE OPTIMUM NUMBER OF CLUSTERS
max_clusters = 10
# Call the function with the PCA-reduced data and the specified maximum number of clusters
optimize_k_means(X_pca, max_clusters)
st.write("We can see that 5 is the optimal value of K")

st.markdown("### APPLYING K- MEANS CLUSTERING FOR DIFFERENT K-VALUES")
ColName='kmeans_'
for k in range(1,6):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_pca[:, :2])
    ColName2=ColName+str(k)
    X_new[ColName2]=kmeans.labels_
    
fig, axs = plt.subplots(nrows=1,ncols=5,figsize=(20,5))
for i, ax in enumerate(fig.axes,start=1):
    ColName2=ColName+str(i)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=X_new[ColName2], cmap='viridis', edgecolor='k', s=50)
    ax.set_title(f'N Cluster : {i}')

st.pyplot(fig)

#3D PLOT
# Applying PCA with 3 principal components
pca_3d = PCA(n_components=3, random_state=50)
X_pca_3d = pca_3d.fit_transform(X_new)

# Plotting 3D scatter plot for the clusters
plot_3d_scatter(X_pca_3d, X_new['kmeans_5'])

#EXTRACTING THE MODEL PARAMETERS
st.markdown("### MODEL PARAMETERS FOR GLOBAL THE K-MEANS CLUSTERING ALGORITHM FOR K =5")
# Fit the K-Means model for K=5
kmeans = KMeans(n_clusters=5)
kmeans.fit(X_pca[:, :2])
# Cluster assignments for each data point
cluster_labels = kmeans.labels_
st.write("Cluster Labels : ",cluster_labels)
# Cluster centers
cluster_centers = kmeans.cluster_centers_
st.write("Cluster Centers : ",cluster_centers)
# Inertia (within-cluster sum of squares)
inertia = kmeans.inertia_
st.write("Inertia within cluster : ",inertia)
# Number of iterations required for convergence
n_iter = kmeans.n_iter_
st.write("Number of Iterations required for convergence : ",n_iter)

#Saving the X_pca in as a csv file.
np.savetxt("F:/SAMYUGTHA/PSGCT/SEMESTER 5/ML/ML CA 3/X_pca.csv", X_pca, delimiter=",")

# Save y_balanced as a CSV file
y_balanced.to_csv("F:/SAMYUGTHA/PSGCT/SEMESTER 5/ML/ML CA 3/y_balanced.csv", index=False)

#Silhoutte Analysis
range_n_clusters = [2, 3, 4, 5]
silhouette_avg_values = []

X = X_pca
silhouette_avg_n_clusters = []

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    silhouette_avg_n_clusters.append(silhouette_avg)
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

style.use("fivethirtyeight")
plt.plot(range_n_clusters, silhouette_avg_n_clusters)
plt.xlabel("Number of Clusters (k)")
plt.ylabel("silhouette score")
plt.show()
st.pyplot(plt)