from sklearn.inspection import permutation_importance

def seleccionar_caracteristicas_por_permutacion(
    modelo,
    X_val,
    y_val,
    top_k
):
    # Calcular importancia por permutación
    resultado = permutation_importance(
        modelo,
        X_val,
        y_val,
        n_repeats=10,
        random_state=42
    )

    # Importancia media de cada característica
    importancias = resultado.importances_mean

    # Índices ordenados de mayor a menor importancia
    indices = importancias.argsort()[::-1]

    # Seleccionar las top_k características
    caracteristicas_top = (
        X_val.columns[indices[:top_k]]
        .tolist()
    )

    return caracteristicas_top
