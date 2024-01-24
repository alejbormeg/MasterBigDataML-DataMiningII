# MasterBigDataML-DataMiningII
 

### Pregunta 3
a) Comentario sobre los gráficos que representan las variables en los planos formados por las componentes:
Las imágenes que muestran las contribuciones de las variables a las componentes principales nos ayudan a entender qué característica física de los pingüinos es capturada por cada componente principal. Generalmente, una componente principal que tiene altas cargas (valores absolutos grandes en sus vectores propios) para ciertas variables significa que esas variables contribuyen significativamente a la variabilidad en esa dirección.

    La Primera Componente Principal parece capturar la mayor parte de la variabilidad asociada con el tamaño general de los pingüinos, ya que variables como la longitud del pico, la profundidad del pico, y la masa corporal tienen grandes contribuciones. Esto sugiere que esta componente podría interpretarse como un factor de "tamaño general" o "masa corporal" de los pingüinos.

    La Segunda Componente Principal podría estar capturando aspectos relacionados con la morfología específica, posiblemente diferenciando entre las proporciones del pico y la longitud de las aletas en relación con el tamaño corporal.

b) Comentario sobre los gráficos que representan las observaciones en los nuevos ejes:
Las especies de pingüinos que se destacan en cada componente se pueden inferir de la posición de las observaciones en el gráfico de dispersión de PCA.

    Las observaciones que tienen valores altos en la Primera Componente Principal son aquellos pingüinos que son grandes en términos de masa corporal y tamaño del pico.

    Las observaciones que tienen valores altos o bajos en la Segunda Componente Principal son aquellos pingüinos que tienen características distintivas de pico y aletas que no están directamente relacionadas con el tamaño general. Por ejemplo, una aleta más larga o un pico más profundo en relación con su masa corporal.

c) Construcción de un índice utilizando una combinación lineal de todas las variables:
Un índice que valore de forma conjunta las características físicas de un pingüino podría construirse tomando los pesos de las cargas de las variables en las componentes principales y sumándolos para cada pingüino. Este índice sería esencialmente una puntuación compuesta basada en las componentes principales que hemos decidido retener.

Para construirlo, podríamos calcular la suma ponderada de las variables estandarizadas para cada pingüino, utilizando los pesos de las cargas de las componentes principales que hemos retenido. Por ejemplo, si retenemos dos componentes principales, el índice para un pingüino podría ser:

Índice_pingüino = (carga_CP1 × valor_variable_1) + (carga_CP2 × valor_variable_2) + ... + (carga_CPn × valor_variable_n)

El valor del índice para una especie de pingüino representada por el conjunto de datos sería el promedio de los índices de todos los pingüinos dentro de esa especie.

    Para la especie 'Adelie', calcularíamos el valor del índice utilizando los valores medios de sus características físicas multiplicados por las cargas correspondientes de las componentes principales retenidas.

    Para la especie 'Chinstrap', haríamos lo mismo con los valores medios de las características físicas de los pingüinos de esa especie.

Estos índices nos darían una puntuación que refleja las características físicas predominantes de las especies basadas en las componentes retenidas del PCA. Estos valores serían únicos para cada especie y podrían servir para diferenciar entre ellas basándonos en las características físicas medidas.

Para realizar estos cálculos y obtener los valores de índice específicos para las especies mencionadas, necesitaríamos los datos crudos y los pesos de las cargas de las componentes de PCA.