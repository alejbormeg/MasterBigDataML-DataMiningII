import os # Proporciona funciones para interactuar con el sistema operativo.
import pandas as pd # Manipulación y análisis de datos tabulares (filas y columnas).
import numpy as np # Operaciones numéricas y matriciales.
import seaborn as sns # Visualización estadística de datos.
import matplotlib.pyplot as plt # Creación de gráficos y visualizaciones.
import missingno

# Matplotlib es una herramienta versátil para crear gráficos desde cero,
# mientras que Seaborn simplifica la creación de gráficos estadísticos.

from sklearn.decomposition import PCA # Implementación del Análisis de Componentes Principales (PCA).
from sklearn.preprocessing import StandardScaler # Estandarización de datos para análisis estadísticos.
from scipy.spatial.distance import euclidean
from FuncionesMineria2 import *

# Cargando el conjunto de datos de Palmer Penguins
penguins_data = sns.load_dataset("penguins")
print(penguins_data.head())

# Visualización de datos perdidos
missingno.matrix(penguins_data)
plt.show()

print(penguins_data.isna().sum())

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

# Gráfico de dispersión de la longitud y profundidad del pico de los pingüinos
sns.scatterplot(data=penguins_data, x='bill_length_mm', y='bill_depth_mm', hue='species')
plt.title('Gráfico de dispersión de la longitud y profundidad del pico de los pingüinos')
plt.xlabel('Longitud del Pico (mm)')
plt.ylabel('Profundidad del Pico (mm)')
plt.show()

# Pairplot para explorar relaciones entre múltiples características
sns.pairplot(data=penguins_data, hue='species', vars=['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
plt.suptitle('Pairplot de características de los pingüinos', y=1.02)
plt.show()


################# Pregunta 1 ######################################
# Calcula la matriz de correlaciones y su representaci´on gr´afica: ¿Cu´ales son
# las variables m´as correlacionadas de forma inversa entre las caracter´ısticas
# f´ısicas de los ping¨uinos?

# Matriz de correlaciones 
matriz_correlaciones = penguins_data.corr(numeric_only=True)

# Crear una máscara para la matriz triangular superior
mask = np.triu(np.ones_like(matriz_correlaciones, dtype=bool))
matriz_correlaciones[mask] = np.nan

# Crear un mapa de calor para la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlaciones, annot=True, cmap='coolwarm')
plt.title("Matriz de correlacion Pinguinos")
plt.show()

################# Pregunta 2 ######################################
# Realiza un an´alisis de componentes principales (PCA) con los datos es-
# tandarizados, calculando un n´umero adecuado de componentes (m´aximo
# 4): Estudiar los valores de los autovalores obtenidos y las gr´aficas que los
# resumen. ¿Cu´al es el n´umero adecuado de componentes para representar
# eficientemente la variabilidad de las especies de ping¨uinos?

numeric_variables = penguins_data.select_dtypes(['int32', 'int64', 'float32', 'float64']).columns.to_list()
penguins_data_numeric = penguins_data[numeric_variables]

# -------------------------------- Analisis PCA:
    
# Estandarizamos los datos:
# Utilizamos StandardScaler() para estandarizar (normalizar) las variables.
# - StandardScaler calcula la media y la desviación estándar de las variables en 'notas' durante el ajuste.
# - Luego, utiliza estos valores para transformar 'notas' de manera que tengan media 0 y desviación estándar 1.
# - El método fit_transform() realiza ambas etapas de ajuste y transformación en una sola llamada.
# Finalmente, convertimos la salida en un DataFrame usando pd.DataFrame().
penguins_data_numeric_estandarizadas = pd.DataFrame(
    StandardScaler().fit_transform(penguins_data_numeric),  # Datos estandarizados
    columns=['{}_z'.format(variable) for variable in numeric_variables],  # Nombres de columnas estandarizadas
    index=penguins_data_numeric.index  # Índices (etiquetas de filas) del DataFrame
)

# Crea una instancia de Análisis de Componentes Principales (ACP):
# - Utilizamos PCA(n_components=4) para crear un objeto PCA que realizará un análisis de componentes principales.
# - Establecemos n_components en 4 para retener el maximo de las componentes principales (maximo= numero de variables).
pca = PCA(n_components=4)

# Aplicar el Análisis de Componentes Principales (ACP) a los datos estandarizados:
# - Usamos pca.fit(notas_estandarizadas) para ajustar el modelo de ACP a los datos estandarizados.
fit = pca.fit(penguins_data_numeric_estandarizadas)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# Obtener la varianza explicada por cada componente principal como un porcentaje de la varianza total.
var_explicada = fit.explained_variance_ratio_*100

# Calcular la varianza explicada acumulada a medida que se agregan cada componente principal.
var_acumulada = np.cumsum(var_explicada)

# Crear un DataFrame de pandas con los datos anteriores y establecer índice.
data = {'Autovalores': autovalores, 'Variabilidad Explicada': var_explicada, 'Variabilidad Acumulada': var_acumulada}
tabla = pd.DataFrame(data, index=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)]) 

# Imprimir la tabla
print(tabla)

resultados_pca = pd.DataFrame(fit.transform(penguins_data_numeric_estandarizadas), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=penguins_data_numeric_estandarizadas.index)


plot_varianza_explicada(var_explicada, fit.n_components_)

# Por la regla del codo lo mejor son 2 componentes principales que acumulan el 88% de la varianza explicada


############################# Pregunta 3 ######################################
# Crea una instancia de ACP con las dos primeras componentes que nos interesan y aplicar a los datos.
pca = PCA(n_components=2)
fit = pca.fit(penguins_data_numeric_estandarizadas)

# Obtener los autovalores asociados a cada componente principal.
autovalores = fit.explained_variance_

# Obtener los autovectores asociados a cada componente principal y transponerlos.
autovectores = pd.DataFrame(pca.components_.T, 
                            columns = ['Autovector {}'.format(i) for i in range(1, fit.n_components_+1)],
                            index = ['{}_z'.format(variable) for variable in numeric_variables])

# Calculamos las dos primeras componentes principales
resultados_pca = pd.DataFrame(fit.transform(penguins_data_numeric_estandarizadas), 
                              columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
                              index=penguins_data_numeric_estandarizadas.index)

# Añadimos las componentes principales a la base de datos estandarizada.
penguins_data_numeric_estandarizadas_cp = pd.concat([penguins_data_numeric_estandarizadas, resultados_pca], axis=1)


# Cálculo de las correlaciones entre las variables originales y las componentes seleccionadas.
# Guardamos el nombre de las variables del archivo conjunto (variables y componentes).
variables_cp = penguins_data_numeric_estandarizadas_cp.columns

# Calculamos las correlaciones y seleccionamos las que nos interesan (variables contra componentes).
correlacion = pd.DataFrame(np.corrcoef(penguins_data_numeric_estandarizadas.T, resultados_pca.T), 
                           index = variables_cp, columns = variables_cp)

n_variables = fit.n_features_in_
correlaciones_penguins_data_numeric_estandarizadas_cp = correlacion.iloc[:fit.n_features_in_, fit.n_features_in_:]



cos2 = correlaciones_penguins_data_numeric_estandarizadas_cp **2
plot_cos2_heatmap(cos2)
#######################################################################################################
         
plot_corr_cos(fit.n_components, correlaciones_penguins_data_numeric_estandarizadas_cp)

##################################################################################################

plot_cos2_bars(cos2)


#######################################################################################

contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2,autovalores,fit.n_components)
######################################################################################################
           
plot_pca_scatter(pca, penguins_data_numeric_estandarizadas, fit.n_components)

################################################################################
          
plot_pca_scatter_with_vectors(pca, penguins_data_numeric_estandarizadas, fit.n_components, fit.components_)


# Calculando el Índice de Características Físicas (ICF) para cada pingüino

penguins_data_cp = pd.concat([penguins_data_numeric_estandarizadas_cp, penguins_data['species']], axis=1)

# Asumiendo que cada variable tiene una carga asociada con el primer componente principal
cargas = cos2.iloc[:, 0]  # Seleccionando las cargas del primer componente principal

# Calculando el ICF con las nuevas cargas
penguins_data_cp['ICF'] = (
    ((cargas['bill_length_mm_z'] * penguins_data_cp['bill_length_mm_z']) +
    (cargas['bill_depth_mm_z'] * penguins_data_cp['bill_depth_mm_z']) +
    (cargas['flipper_length_mm_z'] * penguins_data_cp['flipper_length_mm_z']) +
    (cargas['body_mass_g_z'] * penguins_data_cp['body_mass_g_z']))**2
)

print(penguins_data_cp['ICF'])

# Recalculando el valor del ICF para un pingüino aleatorio y promedios para Adelie y Chinstrap
random_penguin_icf = penguins_data_cp['ICF'].sample(n=1).iloc[0]
adelie_icf_mean = penguins_data_cp[penguins_data_cp['species'] == 'Adelie']['ICF'].mean()
chinstrap_icf_mean = penguins_data_cp[penguins_data_cp['species'] == 'Chinstrap']['ICF'].mean()
gentoo_icf_mean = penguins_data_cp[penguins_data_cp['species'] == 'Gentoo']['ICF'].mean()

print(f"Indice de pinguino aleatorio: {random_penguin_icf}")
print(f"Indice Chinstrap: {chinstrap_icf_mean}")
print(f"Indice Adelie: {adelie_icf_mean}")
print(f"Indice Gentoo: {gentoo_icf_mean}")

##################################################