import pandas as pd
import numpy as np

def agrupar_generaciones(df):
    # Filtrar usuarios activos recientemente
    df_filtrado = df[df["dias_inactivo"] <= 30].copy()

    # Crear rangos de edad
    bins = [0, 18, 35, 60, 100]
    etiquetas = ["Menor", "Joven", "Adulto", "Mayor"]

    df_filtrado["rango_edad"] = pd.cut(
        df_filtrado["edad"],
        bins=bins,
        labels=etiquetas
    )

    # Promedio de tiempo en la app por rango de edad
    promedios = (
        df_filtrado
        .groupby("rango_edad")["tiempo_en_app"]
        .mean()
    )

    # Convertir a arreglo numpy unidimensional
    return promedios.to_numpy()
