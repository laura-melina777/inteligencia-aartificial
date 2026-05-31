import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def predecir_falla_maquinaria(
    ruta_csv: str,
    test_size: float,
    n_neighbors: int
) -> tuple:

    # 1. Cargar dataset
    df = pd.read_csv(ruta_csv)

    # 2. Manejo de valores nulos
    df = df.dropna()

    # 3. Separar variables
    X = df.drop(columns=["falla"])
    y = df["falla"]

    # 4. División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    # 5. Ajustar n_neighbors si es necesario
    n_neighbors = min(n_neighbors, len(X_train))

    if n_neighbors < 1:
        n_neighbors = 1

    # 6. Entrenar modelo
    modelo = KNeighborsClassifier(
        n_neighbors=n_neighbors
    )

    modelo.fit(X_train, y_train)

    # 7. Predicciones
    predicciones = modelo.predict(X_test)

    # 8. Accuracy
    accuracy = accuracy_score(
        y_test,
        predicciones
    )

    return predicciones, accuracy
