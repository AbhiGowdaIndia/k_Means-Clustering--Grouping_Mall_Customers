# K-Means clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

#Using the Elbow method to find the nukmber of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#Plot the Elbow graph
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#Applying kmeans to the dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#Visualize the clusters
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='r',label='Clusyer 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='g',label='Clusyer 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='b',label='Clusyer 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='c',label='Clusyer 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='m',label='Clusyer 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='k',label='Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()