# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns #visualization
from sklearn.preprocessing import StandardScaler #data normalization
from sklearn.cluster import KMeans #kmeans algorithm
import warnings
warnings.filterwarnings("ignore")
from mpl_toolkits.mplot3d import Axes3D # 3d plot
from termcolor import colored as cl # text customization
plt.rcParams['figure.figsize'] = (20, 10) #for setting a plot size and white grid
sns.set_style('whitegrid')
import pickle

df = pd.read_csv('customer_segmentation_dataset.csv')
df.drop('Unnamed: 0', axis = 1,inplace=True)
df.set_index('Customer Id', inplace=True)

X = df.values
X = np.nan_to_num(X)
sc = StandardScaler()
cluster_data = sc.fit_transform(X)
print(cl('Cluster data samples : ', attrs = ['bold']), cluster_data[:5])

# MODELING
clusters =3
model = KMeans(init = 'k-means++', # Initialization method of the centroids
               n_clusters = clusters, # The number of clusters to form 
               n_init = 12) #Number of times the k-means algorithm will be run with different centroid
model.fit(X)
labels = model.labels_
print(cl(labels[:100], attrs = ['bold']))

df['cluster_num'] = labels
df.groupby('cluster_num').mean()
# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[30.   ,  1.   ,  7.   , 33.   ,  1.165,  7.217,  1.   , 25.4  ]]))
