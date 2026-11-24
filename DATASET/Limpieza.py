import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==============================================================================
# 1. CARGA DEL DATASET (150,000 REGISTROS DE SDSS DR17)
# ==============================================================================
ruta_csv = "Skyserver_SQL.csv"  # <-- Cambia esto por el nombre exacto de tu CSV

# Omitir la fila '#Table1' si SkyServer la incluyó al exportar
with open(ruta_csv, "r") as f:
    primera_linea = f.readline()
skip = 1 if primera_linea.startswith("#") else 0

df = pd.read_csv(ruta_csv, skiprows=skip)
print(f"1. Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")

# ==============================================================================
# 2. LIMPIEZA DE VALORES ATÍPICOS EXTREMOS (OUTLIERS EN RADIOS DE PETROSIAN)
# ==============================================================================
columnas_radios = [
    "petroRad_u",
    "petroRad_g",
    "petroRad_r",
    "petroRad_i",
    "petroRad_z",
    "petroR50_r",
    "petroR90_r",
]

# Filtramos saturaciones extremas (> 100 segundos de arco)
condicion_limpia = np.ones(len(df), dtype=bool)
for col in columnas_radios:
    condicion_limpia &= df[col] < 100.0

df_clean = df[condicion_limpia].copy()
eliminados = len(df) - len(df_clean)
print(
    f"2. Limpieza de outliers: {eliminados} filas saturadas eliminadas. "
    f"Restantes: {len(df_clean)} filas"
)

# ==============================================================================
# 3. SEPARACIÓN DE DESCRIPTORES (X) Y CODIFICACIÓN DE CLASE OBJETIVO (y)
# ==============================================================================
X = df_clean.drop(columns=["class"]).values
y_raw = df_clean["class"].values

encoder = LabelEncoder()
y = encoder.fit_transform(y_raw)
print(
    "3. Clases codificadas:",
    dict(zip(encoder.classes_, range(len(encoder.classes_)))),
)

# ==============================================================================
# 4. PARTICIÓN ESTRATIFICADA: 70% TRAIN / 15% VALIDATION / 15% TEST
# ==============================================================================
# Paso A: Separar 15% para Test (queda 85% en X_temp)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Paso B: Del 85% restante, tomar 17.65% para Validation (0.85 * 0.1765 ≈ 15% del total)
# Resultado exacto: 70% Train / 15% Validation / 15% Test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
)

# ==============================================================================
# 5. ESTANDARIZACIÓN DE LOS 15 DESCRIPTORES (SIN FUGA DE DATOS)
# ==============================================================================
scaler = StandardScaler()

# Ajustar (fit) ÚNICAMENTE con X_train y transformar los 3 subconjuntos
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# También preparamos el conjunto combinado Train + Validation (85%) para el modelo final
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_final_scaled = scaler_full.transform(X_test)

print("\n--- RESUMEN FINAL DE PARTICIÓN (70 / 15 / 15) ---")
print(f"X_train_scaled : {X_train_scaled.shape} (~70% - Entrenamiento de pesos)")
print(f"X_val_scaled   : {X_val_scaled.shape} (~15% - Evaluación Fitness GA+GWO)")
print(f"X_test_scaled  : {X_test_scaled.shape} (~15% - Prueba final intacta)")

# ==============================================================================
# 6. EXPORTACIÓN DE LOS NUEVOS CSV (TRAIN, VALIDATION Y TEST)
# ==============================================================================
# Recuperamos los nombres exactos de las 15 columnas descriptoras
nombres_descriptores = df_clean.drop(columns=["class"]).columns.tolist()


def guardar_subconjunto_csv(X_datos, y_datos, nombre_archivo):
    # Crear DataFrame con los 15 descriptores estandarizados
    df_export = pd.DataFrame(X_datos, columns=nombres_descriptores)

    # Agregar la columna objetivo codificada (0, 1, 2)
    df_export["class"] = y_datos

    # (Opcional) Si prefieres guardar también la etiqueta original en texto:
    # df_export['class_label'] = encoder.inverse_transform(y_datos)

    # Exportar a CSV sin el índice numérico de pandas
    df_export.to_csv(nombre_archivo, index=False)
    print(f"Archivo generado: '{nombre_archivo}' -> {df_export.shape}")


print("\n--- EXPORTANDO ARCHIVOS CSV ESCALADOS ---")
guardar_subconjunto_csv(X_train_scaled, y_train, "sdss_train_70.csv")
guardar_subconjunto_csv(X_val_scaled, y_val, "sdss_val_15.csv")
guardar_subconjunto_csv(X_test_scaled, y_test, "sdss_test_15.csv")

# (Opcional) Exportar también el conjunto combinado Train + Validation (85%)
# para entrenar el modelo definitivo una vez terminado el GA + GWO:
guardar_subconjunto_csv(
    X_train_full_scaled, y_train_full, "sdss_train_full_85.csv"
)