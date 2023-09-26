import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
# Genera observaciones normales
np.random.seed(42)
X_normal = 0.3 * np.random.randn(200, 2)

# Genera algunas anomalías
X_anomalies = np.random.uniform(low=-5, high=5, size=(10, 2))

# Combina los datos normales y las anomalías
X = np.vstack([X_normal, X_anomalies])
# Crea el modelo de Isolation Forest
clf = IsolationForest(contamination=0.1, random_state=42)

# Entrena el modelo
clf.fit(X)



# Predice si cada observación es una anomalía o no (1 para normal, -1 para anomalía)
y_pred = clf.predict(X)

# Muestra las etiquetas de anomalía (-1) y normal (1)
print("Etiquetas de anomalía y normal:")
print(y_pred)
# Visualiza los datos y las anomalías detectadas
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title("Isolation Forest - Detección de Anomalías")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()
