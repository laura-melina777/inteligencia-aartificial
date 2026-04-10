import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

def generar_caso_de_uso_imputar_mediciones_iterativamente():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(10, 16))
    columnas_numericas = ["ph", "oxigeno_disuelto", "turbidez", "temperatura"]

    data = {
        "ph": rng.normal(7, 0.5, n_filas),
        "oxigeno_disuelto": rng.normal(8, 1.2, n_filas),
        "turbidez": rng.normal(15, 4, n_filas),
        "temperatura": rng.normal(22, 3, n_filas)
    }

    df = pd.DataFrame(data)

    for col in columnas_numericas:
        mascara = rng.random(n_filas) < 0.2
        df.loc[mascara, col] = np.nan

    input_data = {
        "df": df.copy(),
        "columnas_numericas": columnas_numericas
    }

    X = df[columnas_numericas].copy()
    imputer = IterativeImputer(random_state=42)
    output_data = imputer.fit_transform(X)

    return input_data, output_data
