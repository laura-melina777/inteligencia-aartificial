import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

def detectar_anomalias_red(df):
    """
    Detecta anomalías en tráfico de red usando IsolationForest.

    Argumentos:
        df (pd.DataFrame): DataFrame con columnas:
            duracion, bytes_enviados, bytes_recibidos,
            num_conexiones, puerto_destino

    Retorna:
        dict:
        {
            'df_resultado': pd.DataFrame,
            'porcentaje_anomalias': float,
            'modelo': IsolationForest
        }
    """

    # 1. Copia del dataframe
    df_resultado = df.copy()

    # 2. Variables numéricas requeridas
    cols = [
        'duracion',
        'bytes_enviados',
        'bytes_recibidos',
        'num_conexiones',
        'puerto_destino'
    ]

    X = df_resultado[cols]

    # 3. Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. Modelo Isolation Forest
    modelo = IsolationForest(
        random_state=42,
        contamination='auto'
    )

    predicciones = modelo.fit_predict(X_scaled)

    # 5. Marcar anomalías
    df_resultado['anomalia'] = predicciones == -1

    # 6. Calcular porcentaje
    porcentaje_anomalias = (
        df_resultado['anomalia'].mean() * 100
    )

    return {
        'df_resultado': df_resultado,
        'porcentaje_anomalias': porcentaje_anomalias,
        'modelo': modelo
    }


# Prueba local
if __name__ == "__main__":

    # Aquí usarías el generador del compañero
    entrada, salida_esperada = generar_caso_de_uso_detectar_anomalias_red()

    resultado = detectar_anomalias_red(entrada['df'])

    print(resultado['df_resultado'].head())
    print(
        f"Porcentaje anomalías: "
        f"{resultado['porcentaje_anomalias']:.2f}%"
    )
