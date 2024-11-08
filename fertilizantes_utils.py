import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss

import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

# Filtrar las advertencias de interpolación
warnings.filterwarnings("ignore", category=InterpolationWarning)

def evaluar_estacionareidad(df):

    # Ejecutamos adfuller() en cada columna del conjunto de datos y guardamos los valores-p:
    adfuller_p_values = []
    kpss_p_values = []

    for col in df.columns:

        # Ejecutar la prueba de Dickey-Fuller:
        adfuller_p_value = adfuller(df[col])[1]
        adfuller_p_values.append(adfuller_p_value)

        # Realizar la prueba KPSS:
        kpss_stat, kpss_p_value, lags, crit_values = kpss(df[col])
        kpss_p_values.append(kpss_p_value)

    # Creamos un DataFrame que albergue loos valores-p de la prueba:
    df_test_estacionareidad = pd.DataFrame(
        {'variable': df.columns,
        'valor_p_adfuller': adfuller_p_values,
        'valor_p_kpss': kpss_p_values}
    )

    # Verificamos si los valores-p son inferiores al umbral de significancia:
    df_test_estacionareidad['adfuller_rechazar_H0'] = df_test_estacionareidad['valor_p_adfuller'] < 0.05
    df_test_estacionareidad['kpss_rechazar_H0'] = df_test_estacionareidad['valor_p_kpss'] < 0.05

    resultado_adfuller = df_test_estacionareidad['adfuller_rechazar_H0']
    resultado_kpss = df_test_estacionareidad['kpss_rechazar_H0']

    # # Restaurar la configuración original de advertencias después de ejecutar tu código
    # warnings.filterwarnings("default", category=InterpolationWarning)

    # Crear una columna que indique 'estacionario', 'no estacionario' o 'No claro'
    df_test_estacionareidad['estado_estacionario'] = 'No claro'
    df_test_estacionareidad.loc[resultado_adfuller & ~resultado_kpss, 'estado_estacionario'] = 'Estacionario'
    df_test_estacionareidad.loc[~resultado_adfuller & resultado_kpss, 'estado_estacionario'] = 'No estacionario'

    return df_test_estacionareidad

# ================================================================================================================= #

def apply_outlier_capping(series, multiplier=1.5):
    """
    Aplica capping a los valores atípicos de una serie.

    Parámetros:
    - series (pd.Series): Serie de entrada.
    - multiplier (float): Multiplicador para el IQR para determinar los límites de capping. Default es 1.5.

    Devuelve:
    - pd.Series: Serie con capping aplicado.
    """

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # Definir límites
    upper_limit = Q3 + multiplier * IQR
    lower_limit = Q1 - multiplier * IQR

    # Aplicar capping
    series_capped = series.clip(lower=lower_limit, upper=upper_limit)

    return pd.Series(series_capped)

# ================================================================================================================= #

def cap_outliers_by_percentile(series, lower_percentile=0.05, upper_percentile=0.95):
    """
    Aplica capping a los valores atípicos de una serie según percentiles específicos.

    Parámetros:
    - series (pd.Series): Serie de entrada.
    - lower_percentile (float): Percentil inferior para el capping (valor entre 0 y 1).
    - upper_percentile (float): Percentil superior para el capping (valor entre 0 y 1).

    Devuelve:
    - pd.Series: Serie con capping aplicado.
    """
    
    # Calcular los valores en los percentiles especificados
    lower_limit = series.quantile(lower_percentile)
    upper_limit = series.quantile(upper_percentile)

    # Aplicar capping
    series_capped = series.clip(lower=lower_limit, upper=upper_limit)
    
    return series_capped


# ================================================================================================================= #
from unidecode import unidecode
import re

# def preprocess_text(text, keep_characters=''):
#     """Preprocesamiento básico de texto.
#     Args:
#         text (str): El texto a preprocesar.
#         keep_characters (str): Cadena de caracteres no-alfanuméricos que se desean conservar.
#     Returns:
#         str: El texto preprocesado.
#     """
#     if isinstance(text, str):
#         text = text.lower()
#         text = unidecode(text)

#         # Crea una expresión regular que incluye los caracteres que se quieren conservar.
#         regex_pattern = r'[^a-z0-9\s' + re.escape(keep_characters) + ']'
#         text = re.sub(regex_pattern, ' ', text)

#         text = re.sub(r'\s+', ' ', text).strip()
#     return text

import re
import string
from unidecode import unidecode

def preprocess_text(text, keep_characters='', keep_accents=False):
    """Preprocesamiento avanzado de texto que elimina caracteres no imprimibles, con opción para conservar acentos.
    
    Args:
        text (str): El texto a preprocesar.
        keep_characters (str): Cadena de caracteres no-alfanuméricos que se desean conservar.
        keep_accents (bool): Indica si se deben conservar los acentos en las vocales.
        
    Returns:
        str: El texto preprocesado.
    """
    if isinstance(text, str):
        # Convertir a minúsculas
        text = text.lower()
        
        # Eliminar acentos si keep_accents es False
        if not keep_accents:
            text = unidecode(text)
        
        # Crear una expresión regular que incluya solo caracteres imprimibles y los que se desean conservar
        regex_pattern = r'[^a-z0-9áéíóúñü\s' + re.escape(keep_characters) + ']'
        text = re.sub(regex_pattern, ' ', text)
        
        # Eliminar caracteres no imprimibles de todo el texto, incluidos caracteres de control
        text = ''.join(char for char in text if char in string.printable or char in 'áéíóúñü')
        
        # Eliminar espacios extra
        text = re.sub(r'\s+', ' ', text).strip()
        
    return text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ccf_subplots(df, X_columns, target_var, max_lag=30, palette_name="tab20", tick_fontsize=10, confidence_level=0.95):
    """
    Grafica la Correlación Cruzada Función (CCF) entre múltiples variables independientes y una variable objetivo.

    Parámetros:
    - df (pd.DataFrame): DataFrame que contiene las series temporales de las variables independientes y la variable objetivo.
    - X_columns (list): Lista de nombres de las columnas que son las variables independientes.
    - target_var (str): Nombre de la columna que es la variable objetivo.
    - max_lag (int): Número máximo de lags a considerar para la CCF. Por defecto es 30.
    - palette_name (str): Nombre de la paleta de colores de Seaborn a utilizar. Por defecto es "tab20".
    - tick_fontsize (int): Tamaño de fuente de los ticks en los ejes. Por defecto es 10.
    - confidence_level (float): Nivel de confianza para las líneas de significancia (e.g., 0.95 para 95%). Por defecto es 0.95.

    Retorna:
    - None: Muestra el gráfico de subplots.
    """

    # Número de características (variables independientes)
    num_features = len(X_columns)

    # Determinar el número de filas y columnas para la cuadrícula de subplots
    num_rows = int(np.ceil(np.sqrt(num_features)))  # Calcula el número de filas
    num_cols = int(np.ceil(num_features / num_rows))  # Calcula el número de columnas

    # Seleccionar una paleta de colores de Seaborn con suficientes colores
    palette = sns.color_palette(palette_name, n_colors=num_features)

    # Crear una cuadrícula de subplots con el tamaño adecuado
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5*num_cols, 4*num_rows))

    # Aplanar la lista de ejes para facilitar la iteración
    axes = axes.flatten()

    # Calcular el tamaño de la muestra para los intervalos de confianza
    # Asumimos que la CCF en lag 0 tiene el mayor número de observaciones
    N = df[[X_columns[0], target_var]].dropna().shape[0]

    # Calcular el factor crítico para el nivel de confianza deseado
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - confidence_level) / 2)  # Valor z para el intervalo de confianza

    # Calcular el intervalo de confianza
    ic = z / np.sqrt(N)

    # Iterar sobre cada variable independiente y graficar su CCF con la variable objetivo
    for i, col in enumerate(X_columns):
        ax = axes[i]  # Seleccionar el subplot actual

        # Seleccionar datos sin NaN para la variable actual y la variable objetivo
        df_plot = df[[col, target_var]].dropna()

        # Inicializar listas para almacenar los valores de CCF y los lags correspondientes
        ccf_values = []
        lags = range(0, max_lag + 1)

        # Calcular la CCF manualmente para los lags de 0 a max_lag
        for lag in lags:
            if lag == 0:
                # Correlación sin desplazamiento (lag 0)
                corr = df_plot[col].corr(df_plot[target_var])
            else:
                # Correlación con desplazamiento en el tiempo (lag)
                corr = df_plot[col].corr(df_plot[target_var].shift(lag))
            ccf_values.append(corr)

        # Convertir lags y ccf_values a listas limpias (sin NaN)
        lags_clean = list(lags)
        ccf_clean = [val for val in ccf_values]

        # Graficar la CCF usando un gráfico de barras con el color seleccionado de la paleta
        sns.barplot(x=lags_clean, y=ccf_clean, ax=ax, color=palette[i % len(palette)])

        # Añadir una línea horizontal en y=0 para referencia
        ax.axhline(0, color='black', linewidth=0.8)

        # Añadir líneas de significancia
        ax.axhline(ic, color='red', linestyle='--', linewidth=1, label=f'IC {int(confidence_level*100)}%')
        ax.axhline(-ic, color='red', linestyle='--', linewidth=1)

        # Establecer título y etiquetas de los ejes
        ax.set_title(f'CCF: {col} vs {target_var}', fontsize=12)
        ax.set_xlabel('Lag', fontsize=8)
        ax.set_ylabel('CCF', fontsize=8)

        # Ajustar el tamaño de fuente de los ticks en los ejes
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        # Opcional: Ajustar límites del eje y para una mejor visualización
        ax.set_ylim(-1, 1)

        # Añadir leyenda solo una vez
        if i == 0:
            ax.legend()

    # Eliminar cualquier subplot adicional que no se use (si la cuadrícula tiene más subplots que variables)
    for j in range(num_features, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el diseño para evitar solapamientos y mejorar la presentación
    plt.tight_layout()

    # Mostrar el gráfico final con todos los subplots de CCF
    plt.show()
    
# ================================================================================================================= #

def plot_comparison(df_original, df_nueva, suffix='new'):
    # Definir el número de filas para organizar los subplots en dos columnas
    n_rows = int(np.ceil(len(df_original.columns) / 2))
    
    # Graficar las series originales y las versiones nuevas en subplots de dos columnas
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, n_rows * 3), sharex=True)
    axes = axes.flatten()  # Aplanar la matriz de ejes para facilitar la iteración
    
    for i, col in enumerate(df_original.columns):
        # Graficar la serie original con línea punteada
        axes[i].plot(df_original[col], label=col, color='#4B4B4B', alpha=0.5, linestyle='--')
        
        # Graficar la nueva versión de la serie con línea sólida
        axes[i].plot(df_nueva[col], label=f'{col}{suffix}', color='#1B3A6F')
        
        # Configuración de títulos y leyendas
        axes[i].set_title(col)
        axes[i].legend(loc='upper right')
    
    # Eliminar cualquier subplot vacío si el número de series es impar
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    # Ajustar el layout y mostrar
    plt.tight_layout()
    plt.xlabel("Tiempo")
    plt.show()


# ================================================================================================================= #

import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df, n_rows, bins=10, color_palette='husl'):
    """
    Genera un gráfico de histogramas para cada columna en el DataFrame, con colores distintos.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene los datos a graficar.
        n_rows (int): Número de filas para organizar los subplots.
        bins (int): Número de bins para los histogramas (por defecto es 10).
    """
    # Crear la figura y los subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, 3 * n_rows))

    # Aplanar los ejes para facilitar la iteración
    axes = axes.flatten()

    # Usar una paleta de colores de seaborn para colores automáticos
    colores = sns.color_palette(color_palette, len(df.columns))  # Paleta "husl" para diferentes colores

    # Iterar sobre las columnas de df para graficar cada histograma
    for i, (ax, column, color) in enumerate(zip(axes, df.columns, colores)):
        
        # Graficar el histograma de cada variable con su propio color
        df[column].plot(
            kind='hist',
            bins=bins,  # Ajusta el número de bins si es necesario
            ax=ax,
            color=color,
            alpha=0.7,
            title=column
        )
        
        # Ajustar los límites del eje X basados en los datos de cada variable
        ax.set_xlim([df[column].min(), df[column].max()])

    # Eliminar los subplots vacíos si hay más subplots que variables
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajustar el diseño para evitar superposiciones
    plt.tight_layout()
    plt.show()

# ================================================================================================================= #

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(df, lags=50, color_palette='husl'):
    """
    Genera gráficos ACF y PACF para cada columna de un DataFrame, con colores distintos para cada par de gráficos.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las series temporales a analizar.
        lags (int): Número de lags para los gráficos de ACF y PACF (por defecto es 50).
        color_palette (str): Paleta de colores de seaborn para diferenciar cada par de gráficos (por defecto es 'husl').
    """
    # Número de variables (columnas) en el DataFrame
    n_vars = df.shape[1]

    # Crear una cuadrícula de subgráficos
    fig, axes = plt.subplots(nrows=n_vars, ncols=2, figsize=(12, 2.5 * n_vars))

    # Generar una paleta de colores con tantos colores como columnas tenga el DataFrame
    colores = sns.color_palette(color_palette, n_vars)

    # Iterar sobre cada columna del DataFrame para graficar ACF y PACF
    for i, (column, color) in enumerate(zip(df.columns, colores)):
        # Gráfico ACF
        plot_acf(df[column], lags=lags, alpha=0.05, ax=axes[i, 0])
        axes[i, 0].set_title(f"ACF de {column}", fontsize=10)
        
        # Cambiar el color de las barras y los puntos en el ACF
        for line in axes[i, 0].get_lines():
            line.set_color(color)  # Cambiar el color de los puntos
        for bar in axes[i, 0].collections:
            bar.set_color(color)   # Cambiar el color de las barras

        # Gráfico PACF
        plot_pacf(df[column], lags=lags, alpha=0.05, ax=axes[i, 1])
        axes[i, 1].set_title(f"PACF de {column}", fontsize=10)
        
        # Cambiar el color de las barras y los puntos en el PACF
        for line in axes[i, 1].get_lines():
            line.set_color(color)  # Cambiar el color de los puntos
        for bar in axes[i, 1].collections:
            bar.set_color(color)   # Cambiar el color de las barras

    # Ajustar el diseño para evitar superposiciones
    plt.tight_layout()
    plt.show()
