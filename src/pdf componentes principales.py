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


# Representacion de la variabilidad explicada:   

def plot_varianza_explicada(var_explicada, n_components):
    """
    Representa la variabilidad explicada 
    Args:
      var_explicada (array): Un array que contiene el porcentaje de varianza explicada
        por cada componente principal. Generalmente calculado como
        var_explicada = fit.explained_variance_ratio_ * 100.
      n_components (int): El número total de componentes principales.
        Generalmente calculado como fit.n_components.
    """  
    # Crear un rango de números de componentes principales de 1 a n_components
    num_componentes_range = np.arange(1, n_components + 1)

    # Crear una figura de tamaño 8x6
    plt.figure(figsize=(8, 6))

    # Trazar la varianza explicada en función del número de componentes principales
    plt.plot(num_componentes_range, var_explicada, marker='o')

    # Etiquetas de los ejes x e y
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada')

    # Título del gráfico
    plt.title('Variabilidad Explicada por Componente Principal')

    # Establecer las marcas en el eje x para que coincidan con el número de componentes
    plt.xticks(num_componentes_range)

    # Mostrar una cuadrícula en el gráfico
    plt.grid(True)

    # Agregar barras debajo de cada punto para representar el porcentaje de variabilidad explicada
    # - 'width': Ancho de las barras de la barra. En este caso, se establece en 0.2 unidades.
    # - 'align': Alineación de las barras con respecto a los puntos en el eje x. 
    #   'center' significa que las barras estarán centradas debajo de los puntos.
    # - 'alpha': Transparencia de las barras. Un valor de 0.7 significa que las barras son 70% transparentes.
    plt.bar(num_componentes_range, var_explicada, width=0.2, align='center', alpha=0.7)

    # Mostrar el gráfico
    plt.show()
    
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


  
#####################################################################################################
def plot_cos2_heatmap(cosenos2):
    """
    Genera un mapa de calor (heatmap) de los cuadrados de las cargas en las Componentes Principales (cosenos al cuadrado).

    Args:
        cosenos2 (pd.DataFrame): DataFrame de los cosenos al cuadrado, donde las filas representan las variables y las columnas las Componentes Principales.

    """
    # Crea una figura de tamaño 8x8 pulgadas para el gráfico
    plt.figure(figsize=(8, 8))

    # Utiliza un mapa de calor (heatmap) para visualizar 'cos2' con un solo color
    sns.heatmap(cosenos2, cmap='Blues', linewidths=0.5, annot=False)

    # Etiqueta los ejes (puedes personalizar los nombres de las filas y columnas si es necesario)
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Cuadrados de las Cargas en las Componentes Principales')

    # Muestra el gráfico
    plt.show()

cos2 = correlaciones_penguins_data_numeric_estandarizadas_cp **2
plot_cos2_heatmap(cos2)
#######################################################################################################

        
def plot_corr_cos(n_components, correlaciones_datos_con_cp):
    """
    Genera un gráficos en los que se representa un vector por cada variable, usando como ejes las componentes, la orientación
    y la longitud del vector representa la correlación entre cada variable y dos de las componentes. El color representa el
    valor de la suma de los cosenos al cuadrado.

    Args:
        n_components (int): Número entero que representa el número de componentes principales seleccionadas.
        correlaciones_datos_con_cp (DataFrame): DataFrame que contiene la matriz de correlaciones entre variables y componentes
    """
    # Definir un mapa de color (cmap) sensible a las diferencias numéricas

    cmap = plt.get_cmap('coolwarm')  # Puedes ajustar el cmap según tus preferencias


    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los cosenos al cuadrado
            sum_cos2 = correlaciones_datos_con_cp.iloc[:, i] ** 2 + correlaciones_datos_con_cp.iloc[:, j] ** 2

            # Crear un nuevo gráfico para cada par de componentes principales
            plt.figure(figsize=(10, 10))

            # Dibujar un círculo de radio 1
            circle = plt.Circle((0, 0), 1, fill=False, color='b', linestyle='dotted')

            plt.gca().add_patch(circle)

            # Dibujar vectores para cada variable con colores basados en la suma de los cosenos al cuadrado
            for k, var_name in enumerate(correlaciones_datos_con_cp.index):
                x = correlaciones_datos_con_cp.iloc[k, i]  # Correlación en la primera dimensión
                y = correlaciones_datos_con_cp.iloc[k, j]  # Correlación en la segunda dimensión

                # Seleccionar un color de acuerdo a la suma de los cosenos al cuadrado
                color = cmap(sum_cos2[k])

                # Dibujar el vector con el color seleccionado
                plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=color)

                # Agregar el nombre de la variable junto a la flecha con el mismo color
                plt.text(x, y, var_name, color=color, fontsize=12, ha='right', va='bottom')

            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')

            # Establecer los límites del gráfico
            plt.xlim(-1.1, 1.1)
            plt.ylim(-1.1, 1.1)

            # Agregar un mapa de color (colorbar) y su leyenda
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])  # Evita errores de escala
            plt.colorbar(mappable=sm, orientation='vertical', label='cos^2')  # Agrega la leyenda
            # Mostrar el gráfico
            plt.grid()
            plt.show()
            
plot_corr_cos(fit.n_components, correlaciones_penguins_data_numeric_estandarizadas_cp)


##################################################################################################

def plot_cos2_bars(cos2):
    """
    Genera un gráfico de barras para representar la varianza explicada de cada variable utilizando los cuadrados de las cargas (cos^2).

    Args:
        cos2 (pd.DataFrame): DataFrame que contiene los cuadrados de las cargas de las variables en las componentes principales.

    Returns:
        None
    """
    # Crea una figura de tamaño 8x6 pulgadas para el gráfico
    plt.figure(figsize=(8, 6))

    # Crea un gráfico de barras para representar la varianza explicada por cada variable
    sns.barplot(x=cos2.sum(axis=1), y=cos2.index, color="blue")

    # Etiqueta los ejes
    plt.xlabel('Suma de los $cos^2$')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Varianza Explicada de cada Variable por las Componentes Principales')

    # Muestra el gráfico
    plt.show()
    

plot_cos2_bars(cos2)

#########################################################################################################



#######################################################################################

def plot_contribuciones_proporcionales(cos2, autovalores, n_components):
    """
    Cacula las contribuciones de cada variable a las componentes principales y
    Genera un gráfico de mapa de calor con los datos
    Args:
        cos2 (DataFrame): DataFrame de los cuadrados de las cargas (cos^2).
        autovalores (array): Array de los autovalores asociados a las componentes principales.
        n_components (int): Número de componentes principales seleccionadas.
    """
    # Calcula las contribuciones multiplicando cos2 por la raíz cuadrada de los autovalores
    contribuciones = cos2 * np.sqrt(autovalores)

    # Inicializa una lista para las sumas de contribuciones
    sumas_contribuciones = []

    # Calcula la suma de las contribuciones para cada componente principal
    for i in range(n_components):
        nombre_componente = f'Componente {i + 1}'
        suma_contribucion = np.sum(contribuciones[nombre_componente])
        sumas_contribuciones.append(suma_contribucion)

    # Calcula las contribuciones proporcionales dividiendo por las sumas de contribuciones
    contribuciones_proporcionales = contribuciones.div(sumas_contribuciones, axis=1) * 100

    # Crea una figura de tamaño 8x8 pulgadas para el gráfico
    plt.figure(figsize=(8, 8))

    # Utiliza un mapa de calor (heatmap) para visualizar las contribuciones proporcionales
    sns.heatmap(contribuciones_proporcionales, cmap='Blues', linewidths=0.5, annot=False)

    # Etiqueta los ejes (puedes personalizar los nombres de las filas y columnas si es necesario)
    plt.xlabel('Componentes Principales')
    plt.ylabel('Variables')

    # Establece el título del gráfico
    plt.title('Contribuciones Proporcionales de las Variables en las Componentes Principales')

    # Muestra el gráfico
    plt.show()
    
    # Devuelve los DataFrames de contribuciones y contribuciones proporcionales
    return contribuciones_proporcionales

contribuciones_proporcionales = plot_contribuciones_proporcionales(cos2,autovalores,fit.n_components)
######################################################################################################
def plot_pca_scatter(pca, datos_estandarizados, n_components):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados.

    Args:
        pca (PCA): Objeto PCA previamente ajustado.
        datos_estandarizados (pd.DataFrame): DataFrame de datos estandarizados.
        n_components (int): Número de componentes principales seleccionadas.
    """
    # Representamos las observaciones en cada par de componentes seleccionadas
    componentes_principales = pca.transform(datos_estandarizados)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los valores al cuadrado para cada variable
            # Crea un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
            plt.scatter(componentes_principales[:, i], componentes_principales[:, j])
            
            # Añade etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_estandarizados.index)
    
            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales[k, i], componentes_principales[k, j]))
            
            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')
            
            # Establece el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones en PCA')
            
            plt.show()
            
plot_pca_scatter(pca, penguins_data_numeric_estandarizadas, fit.n_components)
################################################################################




def plot_pca_scatter_with_vectors(pca, datos_estandarizados, n_components, components_):
    """
    Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados
    con vectores de las correlaciones escaladas entre variables y componentes

    Args:
        pca (PCA): Objeto PCA previamente ajustado.
        datos_estandarizados (pd.DataFrame): DataFrame de datos estandarizados.
        n_components (int): Número de componentes principales seleccionadas.
        components_: Array con las componentes.
    """
    # Representamos las observaciones en cada par de componentes seleccionadas
    componentes_principales = pca.transform(datos_estandarizados)
    
    for i in range(n_components):
        for j in range(i + 1, n_components):  # Evitar pares duplicados
            # Calcular la suma de los valores al cuadrado para cada variable
            # Crea un gráfico de dispersión de las observaciones en las dos primeras componentes principales
            plt.figure(figsize=(8, 6))  # Ajusta el tamaño de la figura si es necesario
            plt.scatter(componentes_principales[:, i], componentes_principales[:, j])
            
            # Añade etiquetas a las observaciones
            etiquetas_de_observaciones = list(datos_estandarizados.index)
    
            for k, label in enumerate(etiquetas_de_observaciones):
                plt.annotate(label, (componentes_principales[k, i], componentes_principales[k, j]))
            
            # Dibujar líneas discontinuas que representen los ejes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)
            
            # Etiquetar los ejes
            plt.xlabel(f'Componente Principal {i + 1}')
            plt.ylabel(f'Componente Principal {j + 1}')
            
            # Establece el título del gráfico
            plt.title('Gráfico de Dispersión de Observaciones y variables en PCA')
            
            
            # Añadimos vectores que representen las correlaciones escaladas entre variables y componentes
            fit = pca.fit(datos_estandarizados)
            coeff = np.transpose(fit.components_)
            scaled_coeff = 8 * coeff  #8 = escalado utilizado, ajustar en función del ejemplo
            for var_idx in range(scaled_coeff.shape[0]):
                plt.arrow(0, 0, scaled_coeff[var_idx, i], scaled_coeff[var_idx, j], color='red', alpha=0.5)
                plt.text(scaled_coeff[var_idx, i], scaled_coeff[var_idx, j],
                     penguins_data_numeric_estandarizadas.columns[var_idx], color='red', ha='center', va='center')
            
            plt.show()
            
plot_pca_scatter_with_vectors(pca, penguins_data_numeric_estandarizadas, fit.n_components, fit.components_)

    
##################################################

















































# # Cargar un archivo Excel llamado 'notas.xlsx' en un DataFrame llamado notas.
# notas_S = pd.read_excel('notas_S.xlsx') 

# # Establecer la columna 'Alumno' como índice del DataFrame notas y eliminarla.
# notas_S = notas_S.set_index('Alumno', drop=True)

# # Guarda la variable el indice y 'EXTRA_ESC' en un dataframe
# extra_S = notas_S.iloc[:, [7]]

# # Elimina la variable 'EXTRA_ESC' del DataFrame 'notas'.
# notas_S = notas_S.drop(notas_S.columns[7], axis=1)

# # Calcular la media y la desviación estándar de 'notas'
# media_notas = notas.mean()
# desviacion_estandar_notas = notas.std()

# # Estandarizar 'notas_S' utilizando la media y la desviación estándar de 'notas'
# notas_S_estandarizadas = pd.DataFrame(((notas_S - media_notas) / desviacion_estandar_notas))

# notas_S_estandarizadas.columns = ['{}_z'.format(variable) for variable in variables]

# # Agregar las observaciones estandarizadas a 'notas'
# notas_sup = pd.concat([notas_estandarizadas, notas_S_estandarizadas])

# # Calcular las componentes principales para el conjunto de datos combinado
# componentes_principales_sup = pca.transform(notas_sup)

# # Calcular las componentes principales para el conjunto de datos combinado
# # y renombra las componentes
# resultados_pca_sup = pd.DataFrame(fit.transform(notas_sup), 
#                               columns=['Componente {}'.format(i) for i in range(1, fit.n_components_+1)],
#                               index=notas_sup.index)

# # Representacion observaciones + suplementarios
# plot_pca_scatter(pca, notas_sup, fit.n_components)




# ######################################################
# # Añadimos la variable categórica "EXTRA_ESC" en los datos
# notas_componentes_sup= pd.concat([notas_sup, resultados_pca_sup], axis=1)  

# extra_sup = pd.concat([extra, extra_S], axis=0)
# notas_componentes_sup_extra= pd.concat([notas_componentes_sup,
#                                                extra_sup], axis=1)  

# #################################################################################################

# def plot_pca_scatter_with_categories(datos_componentes_sup_var, componentes_principales_sup, n_components, var_categ):
#     """
#     Genera gráficos de dispersión de observaciones en pares de componentes principales seleccionados con categorías.

#     Args:
#         datos_componentes_sup_var (pd.DataFrame): DataFrame que contiene las categorías.
#         componentes_principales_sup (np.ndarray): Matriz de componentes principales.
#         n_components (int): Número de componentes principales seleccionadas.
#         var_categ (str): Nombre de la variable introducida
#     """
#     # Obtener las categorías únicas
#     categorias = datos_componentes_sup_var[var_categ].unique() #Modificar por el nombre de la variable categórica

#     for i in range(n_components):
#         for j in range(i + 1, n_components):  # Evitar pares duplicados
#             # Crear un gráfico de dispersión de las observaciones en las dos primeras componentes principales
#             plt.figure(figsize=(8, 6))  # Ajustar el tamaño de la figura si es necesario
#             plt.scatter(componentes_principales_sup[:, i], componentes_principales_sup[:, j])

#             for categoria in categorias:
#                 # Filtrar las observaciones por categoría
#                 observaciones_categoria = componentes_principales_sup[datos_componentes_sup_var[var_categ] == categoria]
#                 # Calcular el centroide de la categoría
#                 centroide = np.mean(observaciones_categoria, axis=0)
#                 plt.scatter(centroide[i], centroide[j], label=categoria, s=100, marker='o')

#             # Añadir etiquetas a las observaciones
#             etiquetas_de_observaciones = list(datos_componentes_sup_var.index)

#             for k, label in enumerate(etiquetas_de_observaciones):
#                 plt.annotate(label, (componentes_principales_sup[k, i], componentes_principales_sup[k, j]))
#                 # Dibujar líneas discontinuas que representen los ejes

#             # Dibujar líneas discontinuas que representen los ejes
#             plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
#             plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

#             # Etiquetar los ejes
#             plt.xlabel(f'Componente Principal {i + 1}')
#             plt.ylabel(f'Componente Principal {j + 1}')

#             # Establecer el título del gráfico
#             plt.title('Gráfico de Dispersión de Observaciones en PCA')

#             # Mostrar la leyenda para las categorías
#             plt.legend()
#             plt.show()
            
# plot_pca_scatter_with_categories(notas_componentes_sup_extra, componentes_principales_sup, fit.n_components, 'EXTRA_ESC')
