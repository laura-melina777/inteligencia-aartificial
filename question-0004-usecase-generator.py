import numpy as np
import pandas as pd
from sklearn.manifold import Isomap

def generar_caso_de_uso_reducir_movilidad_con_isomap():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(12, 20))
    n_componentes = int(rng.integers(2, 4))
    n_neighbors = int(rng.integers(3, 6))

    df = pd.DataFrame({
        "flujo_vehicular": rng.normal(200, 40, n_filas),
        "velocidad_media": rng.normal(35, 8, n_filas),
        "tiempo_espera": rng.normal(12, 3, n_filas),
        "ocupacion_vial": rng.normal(0.65, 0.1, n_filas)
    })

    input_data = {
        "df": df.copy(),
        "n_componentes": n_componentes,
        "n_neighbors": n_neighbors
    }

    X = df.select_dtypes(include=[np.number])

    modelo = Isomap(
        n_components=n_componentes,
        n_neighbors=n_neighbors
    )
    output_data = modelo.fit_transform(X)

    return input_data, output_data
