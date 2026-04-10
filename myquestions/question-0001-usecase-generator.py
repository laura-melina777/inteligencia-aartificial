import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering

def generar_caso_de_uso_agrupar_clientes_espectralmente():
    # Semilla para reproducibilidad
    seed = np.random.randint(0, 10000)
    rng = np.random.default_rng(seed)

    # Parámetros
    n_clusters = int(rng.integers(2, 4))
    n_por_cluster = int(rng.integers(5, 8))

    # Generar centros
    centros = rng.uniform(-5, 5, size=(n_clusters, 4))
    datos = []

    for i in range(n_clusters):
        grupo = centros[i] + rng.normal(0, 0.4, size=(n_por_cluster, 4))
        datos.append(grupo)

    # Unir datos
    X = np.vstack(datos)
    rng.shuffle(X)

    # DataFrame
    df = pd.DataFrame(
        X,
        columns=[
            "horas_reproduccion",
            "generos_consumidos",
            "sesiones_semana",
            "tiempo_sesion"
        ]
    )

    # INPUT
    input_data = {
        "df": df.copy(),
        "n_clusters": n_clusters
    }

    # PROCESAMIENTO (ground truth)
    X_num = df.select_dtypes(include=[np.number])
    X_scaled = StandardScaler().fit_transform(X_num)

    modelo = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=3,  # 🔥 clave para evitar errores
        assign_labels="kmeans",
        random_state=42
    )

    labels = modelo.fit_predict(X_scaled)

    # OUTPUT
    output_data = np.array(labels)

    return input_data, output_data
