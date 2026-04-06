2) Imputación iterativa de variables numéricas
Pregunta

Un laboratorio ambiental registra datos de calidad del agua en un DataFrame, pero algunas columnas numéricas tienen valores faltantes. Se quiere completar esos valores usando información del resto de variables numéricas.

Escribe una función llamada imputar_mediciones_iterativamente(df, columnas_numericas) que:

seleccione únicamente las columnas listadas en columnas_numericas,
aplique IterativeImputer de sklearn con random_state=42,
devuelva únicamente un np.ndarray con las columnas imputadas, en el mismo orden recibido.
