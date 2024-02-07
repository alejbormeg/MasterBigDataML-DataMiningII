import os # Proporciona funciones para interactuar con el sistema operativo.
import pandas as pd # Manipulación y análisis de datos tabulares (filas y columnas).
import numpy as np # Operaciones numéricas y matriciales.
import seaborn as sns # Visualización estadística de datos.
import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones.
import missingno
import scipy.cluster.hierarchy as sch

# Matplotlib es una herramienta versátil para crear gráficos desde cero,
# mientras que Seaborn simplifica la creación de gráficos estadísticos.

from sklearn.decomposition import PCA # Implementación del Análisis de Componentes Principales (PCA).
from sklearn.preprocessing import StandardScaler # Estandarización de datos para análisis estadísticos.
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from FuncionesMineria2 import *
from scipy . spatial import distance
from sklearn.metrics import silhouette_samples

# Cargando el conjunto de datos de Palmer Penguins
penguins_data = sns.load_dataset("penguins")
print(penguins_data.head())

# Visualización de datos perdidos
missingno.matrix(penguins_data)
plt.show()

# Columnas específicas para verificar NaN
columnas_a_verificar = ['bill_depth_mm', 'bill_length_mm', 'body_mass_g', 'flipper_length_mm']

# Eliminar filas donde todas las columnas especificadas son NaN
penguins_data = penguins_data.dropna(subset=columnas_a_verificar, how='all')

# Imputación del valor de 'sex'
# Calcula las medias para machos y hembras
means_male = penguins_data[penguins_data['sex'] == 'Male'][columnas_a_verificar].mean()
means_female = penguins_data[penguins_data['sex'] == 'Female'][columnas_a_verificar].mean()

# Función para calcular la distancia a las medias
def impute_sex(row):
    if pd.isna(row['sex']):
        dist_to_male = euclidean(row[columnas_a_verificar], means_male)
        dist_to_female = euclidean(row[columnas_a_verificar], means_female)
        return 'Male' if dist_to_male < dist_to_female else 'Female'
    else:
        return row['sex']

# Aplicar la función para imputar 'Sex'
penguins_data['sex'] = penguins_data.apply(impute_sex, axis=1)

# Verificar los datos faltantes después de la imputación
missingno.matrix(penguins_data)
plt.show()

# Calcula las estadísticas descriptivas
estadisticos = penguins_data.describe().transpose()
estadisticos['Datos Perdidos'] = penguins_data.isna().sum()  # Cuenta los valores NaN por variable.
print(estadisticos)


############# Pregunta 4 ################

# Creamos clustering jerárquico
penguins_data_cluster = penguins_data[columnas_a_verificar]

sns.clustermap ( penguins_data_cluster , cmap ='coolwarm', annot = True )
plt.title ('Heatmap Penguins')
plt.xlabel ('Ancho/Longitud de pico mm, Masa corporal g, Longitud aleta en mm')
plt.ylabel ('Id')
# Display the plot
plt.show ()

distancematrix = distance.cdist(penguins_data_cluster, penguins_data_cluster, 'euclidean')
distancesmall = distancematrix[:5 , :5]
distancesmall = pd.DataFrame(distancesmall, index = penguins_data_cluster.index[:5], columns = penguins_data_cluster.index[:5])
distancesmallrounded = distancesmall.round(2)
print ('Matriz de distancias :\n', distancesmallrounded )

#Representación visual de la matriz de distancias
plt.figure(figsize=(8, 6))
df_distances = pd.DataFrame(distancematrix)
sns.heatmap(df_distances, annot=False, cmap="YlGnBu", fmt=".1f")

plt.show()

"""Standarizing the variables"""

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the DataFrame to standardize the columns
df_std = pd.DataFrame(scaler.fit_transform(penguins_data_cluster), columns=penguins_data_cluster.columns)

print(df_std)

# Calculate the pairwise Euclidean distances
distance_std = distance.cdist(df_std, df_std,"euclidean")

print(distance_std[:5,:5].round(2))

"""Recalculamos la matriz de distancias y la representamos con los datos estandarizados."""

plt.figure(figsize=(8, 6))
df_std_distance = pd.DataFrame(distance_std, index = df_std.index, columns = df_std.index)
sns.heatmap(df_std_distance, annot=False, cmap="YlGnBu", fmt=".1f")
plt.show()

# Perform hierarchical clustering to get the linkage matrix
linkage = sns.clustermap(df_std_distance, cmap="YlGnBu", fmt=".1f", annot=False, method='average').dendrogram_row.linkage

# Reorder the data based on the hierarchical clustering
order = pd.DataFrame(linkage, columns=['cluster_1', 'cluster_2', 'distance', 'new_count']).index
reordered_data = penguins_data_cluster.reindex(index=order, columns=order)

# Optionally, you can add color bar
sns.heatmap(reordered_data, cmap="YlGnBu", fmt=".1f", cbar=False)
plt.show()


# Calculate the linkage matrix
linkage_matrix = sch.linkage(df_std_distance, method='ward')  # You can choose a different linkage method if needed

# Create the dendrogram
dendrogram = sch.dendrogram(linkage_matrix, labels=penguins_data_cluster.index, leaf_font_size=9, leaf_rotation=90)

# Add a horizontal line at y=100
plt.axhline(y=150, color='r', linestyle='--')

# Display the dendrogram
plt.show()

# Assign data points to 4 clusters
num_clusters = 3
cluster_assignments = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Display the cluster assignments
print("Cluster Assignments:", cluster_assignments)

# Display the dendrogram
plt.show()

"""# Añadimos la nueva variable a nustro data frame"""

# Create a new column 'Cluster' and assign the 'cluster_assignments' values to it
df_std['Cluster4'] = cluster_assignments

# Now 'df' contains a new column 'Cluster' with the cluster assignments

print(df_std["Cluster4"])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_std)

# Create a new DataFrame for the 2D principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 2: Create a scatter plot with colors for clusters
plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(cluster_assignments):
    plt.scatter(df_pca.loc[cluster_assignments == cluster, 'PC1'],
                df_pca.loc[cluster_assignments == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(df_std.index[i]), fontsize=8)

plt.title("2D PCA Plot with Cluster Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

# Justificar que se elejimos al final 4 grupos por el dendograma

#################################### Pregunta 5 ###################################################################

# Set the number of clusters (k=3)
k = 3

# Initialize the KMeans model
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the KMeans model to your standardized data
kmeans.fit(df_std)

# Get the cluster labels for your data
kmeans_cluster_labels = kmeans.labels_

print(kmeans_cluster_labels)

"""Repetimos el gráfico anterior con el k-means. ¿Será igual el gráfico?"""

# Step 2: Create a scatter plot with colors for clusters
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_std)

# Create a new DataFrame for the 2D principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(kmeans_cluster_labels):
    plt.scatter(df_pca.loc[kmeans_cluster_labels == cluster, 'PC1'],
                df_pca.loc[kmeans_cluster_labels == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(df_std.index[i]), fontsize=8)

plt.title("2D PCA Plot with K-means Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()


#Create an array to store the WCSS values for different values of K:
wcss = []

for k in range(1, 11):  # You can choose a different range of K values
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    wcss.append(kmeans.inertia_)  # Inertia is the WCSS value

"""Plot the WCSS values against the number of clusters (K) and look for the "elbow" point:"""

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-', color='b')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

"""Otro método es el de las siluetas"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Create an array to store silhouette scores for different values of K

silhouette_scores = []

#Run K-means clustering for a range of K values and calculate the silhouette score for each K:

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_std)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(df_std, labels)
    silhouette_scores.append(silhouette_avg)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o', linestyle='-', color='b')
plt.title('Silhouette Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()


##### Por el método del codo entre 3 y 4 está bien y por Silhouette serían 2

############################################ Pregunta 6 ##################################################

"""Run K-means clustering with the optimal number of clusters (determined using the Silhouette Method) and obtain cluster labels for each data point:"""

# Assuming 'df_std_distance' is your standardized data and '4' is the optimal number of clusters
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(df_std)
labels = kmeans.labels_

"""Calculates silouhette scores for each clúster"""

silhouette_values = silhouette_samples(df_std, labels)
silhouette_values

plt.figure(figsize=(8, 6))

y_lower = 10
for i in range(4):
    ith_cluster_silhouette_values = silhouette_values[labels == i]
    ith_cluster_silhouette_values.sort()

    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = plt.cm.get_cmap("Spectral")(float(i) / 4)
    plt.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)

    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10

plt.title("Silhouette Plot for Clusters")
plt.xlabel("Silhouette Coefficient Values")
plt.ylabel("Cluster Label")
plt.grid(True)
plt.show()

"""sort by labels para caracterizar los clusters"""

# Add the labels as a new column to the DataFrame
df_std['label'] = labels
# Sort the DataFrame by the "label" column
df_std_sort = df_std.sort_values(by="label")
df_std_sort['label']

# Group the data by the 'label' column and calculate the mean of each group
cluster_centroids = df_std_sort.groupby('label').mean()
cluster_centroids.round(2)
# 'cluster_centroids' now contains the centroids of each cluster

"""Lo mismo pero con los datos originales"""

# Add the labels as a new column to the DataFrame
penguins_data_cluster['label'] = labels
# Sort the DataFrame by the "label" column
df_sort = penguins_data_cluster.sort_values(by="label")

# Group the data by the 'label' column and calculate the mean of each group
cluster_centroids_orig = df_sort.groupby('label').mean()
cluster_centroids_orig.round(2)
# 'cluster_centroids' now contains the centroids of each cluster


################################# PREGUNTA 8 ###############################
# Paso 1: Aplicar PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_std)

# Paso 2: Crear un DataFrame con los componentes principales
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Paso 3: Unir las etiquetas de especie con el DataFrame de PCA
df_pca = pd.concat([df_pca, penguins_data['species']], axis=1)

# Paso 4: Crear el scatter plot coloreado por especie
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='species', palette='Set1')
plt.title('Gráfica de dispersión por especies')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend(title='Species')
plt.grid(True)
plt.show()


# Set the number of clusters (k=2)
k = 3

# Initialize the KMeans model
kmeans = KMeans(n_clusters=k, random_state=0)

# Fit the KMeans model to your standardized data
kmeans.fit(df_std)

# Get the cluster labels for your data
kmeans_cluster_labels = kmeans.labels_

print(kmeans_cluster_labels)

"""Repetimos el gráfico anterior con el k-means. ¿Será igual el gráfico?"""

# Step 2: Create a scatter plot with colors for clusters
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_std)

# Create a new DataFrame for the 2D principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(kmeans_cluster_labels):
    plt.scatter(df_pca.loc[kmeans_cluster_labels == cluster, 'PC1'],
                df_pca.loc[kmeans_cluster_labels == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(df_std.index[i]), fontsize=8)

plt.title("2D PCA Plot with K-means Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()


# Assign data points to 4 clusters
num_clusters = 3
cluster_assignments = sch.fcluster(linkage_matrix, num_clusters, criterion='maxclust')

# Display the cluster assignments
print("Cluster Assignments:", cluster_assignments)

# Display the dendrogram
plt.show()

"""# Añadimos la nueva variable a nustro data frame"""

# Create a new column 'Cluster' and assign the 'cluster_assignments' values to it
df_std['Cluster4'] = cluster_assignments

# Now 'df' contains a new column 'Cluster' with the cluster assignments

print(df_std["Cluster4"])

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_std)

# Create a new DataFrame for the 2D principal components
df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Step 2: Create a scatter plot with colors for clusters
plt.figure(figsize=(10, 6))

# Loop through unique cluster assignments and plot data points with the same color
for cluster in np.unique(cluster_assignments):
    plt.scatter(df_pca.loc[cluster_assignments == cluster, 'PC1'],
                df_pca.loc[cluster_assignments == cluster, 'PC2'],
                label=f'Cluster {cluster}')
# Add labels to data points
for i, row in df_pca.iterrows():
    plt.text(row['PC1'], row['PC2'], str(df_std.index[i]), fontsize=8)

plt.title("2D PCA Plot with Cluster Assignments")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()


# Paso 3: Unir las etiquetas de especie con el DataFrame de PCA
df_pca = pd.concat([df_pca, penguins_data['sex']], axis=1)

# Paso 4: Crear el scatter plot coloreado por especie
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='sex', palette='Set1')
plt.title('Gráfica de dispersión por sexo')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend(title='Species')
plt.grid(True)
plt.show()


# Paso 3: Unir las etiquetas de especie con el DataFrame de PCA
df_pca = pd.concat([df_pca, penguins_data['island']], axis=1)

# Paso 4: Crear el scatter plot coloreado por especie
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='island', palette='Set1')
plt.title('Gráfica de dispersión por island')
plt.xlabel('Componente principal 1')
plt.ylabel('Componente principal 2')
plt.legend(title='Species')
plt.grid(True)
plt.show()
