# Isolation_Tree
Proyecto de Ejemplo para proyecto de Tesis, Tecnicas de Aprendizaje Automatico en la Ciberseguridad
Este código utiliza la biblioteca scikit-learn para crear y entrenar un modelo de Isolation Forest y detectar anomalías en un conjunto de datos sintético. Vamos a analizarlo paso a paso:

# Paso 1: Importar las bibliotecas necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


•	Importamos las bibliotecas necesarias. numpy se utiliza para la generación de datos aleatorios, matplotlib.pyplot para visualizar los resultados y IsolationForest de sklearn.ensemble para crear el modelo de Isolation Forest.

# Paso 2: Generar un conjunto de datos de ejemplo con observaciones normales y anomalías
np.random.seed(42)
X_normal = 0.3 * np.random.randn(100, 2)
X_anomalies = np.random.uniform(low=-5, high=5, size=(10, 2))
X = np.vstack([X_normal, X_anomalies])



En este paso, generamos un conjunto de datos de ejemplo:
•	X_normal contiene 100 observaciones generadas aleatoriamente a partir de una distribución normal con media cero y desviación estándar de 0.3. Estas son observaciones "normales".
•	X_anomalies contiene 10 observaciones generadas aleatoriamente en un rango entre -5 y 5 en ambas dimensiones. Estas son observaciones "anómalas".
•	X combina ambos conjuntos de datos para formar el conjunto completo.

# Paso 3: Crear el modelo de Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)


Creamos una instancia del modelo de Isolation Forest. contamination es un parámetro que establece la proporción de valores atípicos esperados en el conjunto de datos. En este caso, estamos asumiendo que alrededor del 10% de las observaciones son anomalías. random_state se utiliza para asegurar la reproducibilidad de los resultados.

# Paso 4: Entrenar el modelo
clf.fit(X)




Entrenamos el modelo de Isolation Forest con nuestro conjunto de datos X.

# Paso 5: Realizar predicciones para identificar las anomalías
y_pred = clf.predict(X)


Utilizamos el modelo entrenado para hacer predicciones sobre las observaciones en X. El resultado y_pred contiene etiquetas, donde 1 representa observaciones normales y -1 representa observaciones anómalas.

# Paso 6: Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Isolation Forest - Detección de Anomalías")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()

Finalmente, visualizamos los resultados. Usamos plt.scatter para crear un gráfico de dispersión de las observaciones en función de las dos características, utilizando los resultados de y_pred para colorear las observaciones normales y anómalas de manera diferente. También agregamos un título y etiquetas de ejes al gráfico y mostramos la visualización utilizando plt.show().

En resumen, este código utiliza el Isolation Forest para detectar anomalías en un conjunto de datos ficticio y muestra los resultados en un gráfico de dispersión. Las anomalías se destacan en un color diferente en la visualización. Este es un ejemplo sencillo de cómo usar el Isolation Forest para la detección de anomalías en scikit-learn.
