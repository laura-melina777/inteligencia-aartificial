import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

def generar_caso_de_uso_rankear_por_informacion_mutua():
    rng = np.random.default_rng()

    n_filas = int(rng.integers(15, 25))

    x1 = rng.normal(10, 2, n_filas)
    x2 = rng.normal(20, 3, n_filas)
    x3 = rng.normal(30, 4, n_filas)
    x4 = rng.normal(40, 5, n_filas)

    y = 2.5 * np.sin(x1) + 0.8 * x3 + rng.normal(0, 0.5, n_filas)

    df = pd.DataFrame({
        "humedad_suelo": x1,
        "nitrogeno": x2,
        "temperatura": x3,
        "lluvia": x4,
        "rendimiento": y
    })

    input_data = {
        "df": df.copy(),
        "target_col": "rendimiento"
    }

    df_num = df.select_dtypes(include=[np.number])
    X = df_num.drop(columns=["rendimiento"])
    y_target = df_num["rendimiento"].to_numpy()

    scores = mutual_info_regression(X, y_target, random_state=42)
    orden = np.argsort(scores)[::-1]
    output_data = np.array(X.columns[orden], dtype=str)

    return input_data, output_data
