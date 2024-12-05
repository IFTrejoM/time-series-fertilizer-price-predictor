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

def plot_comparison(df_original, df_nueva, prefix='', suffix=''):
    # Definir el número de filas para organizar los subplots en dos columnas
    n_rows = int(np.ceil(len(df_original.columns) / 2))
    
    # Graficar las series originales y las versiones nuevas en subplots de dos columnas
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, n_rows * 3), sharex=True)
    axes = axes.flatten()  # Aplanar la matriz de ejes para facilitar la iteración
    
    for i, col in enumerate(df_original.columns):
        # Graficar la serie original con línea punteada
        axes[i].plot(df_original[col], label=col, color='#4B4B4B', alpha=0.5, linestyle='--')
        
        # Graficar la nueva versión de la serie con línea sólida
        axes[i].plot(df_nueva[col], label=f'{prefix}{col}{suffix}', color='#1B3A6F')
        
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

# ================================================================================================================= #

def plot_side_by_side_histograms(df_original, df_transformed, transformed_suffix="", bins=20):
    """
    Grafica histogramas en subplots organizados en filas, con el original y el transformado
    uno al lado del otro en cada fila. La variable transformada puede tener un sufijo o prefijo personalizado.
    
    Args:
        df_original (pd.DataFrame): DataFrame con las variables originales.
        df_transformed (pd.DataFrame): DataFrame con las variables transformadas.
        transformed_suffix (str): Sufijo, prefijo, o texto adicional en el nombre de las variables transformadas.
        bins (int): Número de bins para los histogramas (por defecto es 20).
    """
    # Número de variables (columnas)
    n_vars = df_original.shape[1]

    # Crear una cuadrícula de subplots con dos columnas y una fila por variable
    fig, axes = plt.subplots(nrows=n_vars, ncols=2, figsize=(12, 3 * n_vars))

    # Iterar sobre cada columna para graficar los histogramas en la misma fila
    for i, col in enumerate(df_original.columns):
        ax_orig, ax_trans = axes[i] if n_vars > 1 else (axes[0], axes[1])  # Ajuste para un solo subplot
        
        # Histograma de la columna original
        df_original[col].plot(kind='hist', bins=bins, alpha=0.6, color='blue', ax=ax_orig)
        ax_orig.set_title(f"Original - {col}")
        
        # Histograma de la columna transformada, construyendo el nombre con el sufijo o prefijo
        transformed_col_name = transformed_suffix + col if transformed_suffix else col
        df_transformed[transformed_col_name].plot(kind='hist', bins=bins, alpha=0.6, color='orange')

# ================================================================================================================= #

from statsmodels.stats.api import het_breuschpagan, het_goldfeldquandt

def check_homoscedasticity(series):
    """
    Check the homoscedasticity of a time series using the Breusch-Pagan test.
    
    Parameters:
    series (pd.Series): Time series data.
    
    Returns:
    dict: Test results including the LM statistic and p-value.
    """
    # Ensure the series is a pandas Series and drop NaN values
    series = series.dropna()

    # Create a time variable as the predictor
    time = np.arange(len(series)).reshape(-1, 1)

    # Add a constant to the predictors
    exog = np.hstack([np.ones((len(time), 1)), time])

    # Perform the Breusch-Pagan test
    lm_stat, lm_pval, _, _ = het_breuschpagan(series, exog)

    # Return the results as a dictionary
    return round(lm_stat, 3), round(lm_pval, 3)

# ================================================================================================================= #

from statsmodels.tsa.stattools import ccf
from scipy.stats import norm

def get_most_significant_lag(x, y, max_lag=30, confidence_level=0.95):
    """
    Encuentra el lag más significativo entre dos series basado en la CCF.
    
    Parámetros:
    - x (pd.Series o np.array): Serie independiente.
    - y (pd.Series o np.array): Serie dependiente.
    - max_lag (int): Número máximo de lags a analizar. Por defecto es 30.
    - confidence_level (float): Nivel de confianza para evaluar significancia. Por defecto es 0.95.
    
    Retorna:
    - dict: Contiene el lag más significativo, su valor de CCF y si es estadísticamente significativo.
    """
    # Asegurarse de que las series no tengan valores NaN
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Calcular CCF para los lags positivos
    ccf_vals = [np.corrcoef(x[:-lag] if lag > 0 else x, y[lag:] if lag > 0 else y)[0, 1] for lag in range(max_lag + 1)]
    
    # Convertir CCF a un array numpy
    ccf_vals = np.array(ccf_vals)

    # Calcular el intervalo de confianza
    n = len(x)
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    ic = z / np.sqrt(n)
    
    # Encontrar el lag más significativo basado en el valor absoluto de CCF
    most_significant_lag = np.argmax(np.abs(ccf_vals))
    most_significant_ccf = ccf_vals[most_significant_lag]
    
    # Verificar si el lag es estadísticamente significativo
    is_significant = np.abs(most_significant_ccf) > ic
    
    return {
        "lag": most_significant_lag,
        "ccf": most_significant_ccf,
        "is_significant": is_significant,
        "confidence_interval": ic
    }

# ================================================================================================================= #

import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_change_points(df, n_bkps=3, annotate_breakpoints=True):
    """
    Función para graficar rupturas estructurales en series temporales de un dataframe.

    Args:
        df (DataFrame): DataFrame con las series temporales, donde cada columna es una serie.
        n_bkps (int): Número máximo de rupturas significativas a buscar.
        annotate_breakpoints (bool): Si True, agrega anotaciones con las fechas de las rupturas.

    Returns:
        None. Genera un gráfico con las series y sus rupturas.
    """
    series_to_analyze = df.columns
    colors = ["red", "green"]  # Colores alternos para sombreado

    n_rows = len(series_to_analyze)
    fig, axes = plt.subplots(n_rows, 1, figsize=(8, n_rows * 3))

    if n_rows == 1:  # Asegurar que axes sea iterable si solo hay una serie
        axes = [axes]

    for i, variable in enumerate(series_to_analyze):
        serie = df[variable].dropna()
        algo = rpt.Binseg(model="l2").fit(serie.values)
        breakpoints = algo.predict(n_bkps=n_bkps)
        
        # Graficar la serie
        axes[i].plot(serie.index, serie, label=f"{variable}", color="blue")
        
        # Sombrear áreas y opcionalmente agregar anotaciones
        for j in range(len(breakpoints) - 1):
            start = serie.index[breakpoints[j]]
            end = serie.index[breakpoints[j + 1] - 1]
            color = colors[j % len(colors)]
            axes[i].axvspan(start, end, color=color, alpha=0.2)
            
            if annotate_breakpoints:
                date_label = start.strftime('%Y-%m-%d')
                axes[i].text(start, serie.iloc[breakpoints[j]], date_label, color="gray",
                             fontsize=8, rotation=90)

        # Configurar eje X y título
        axes[i].xaxis.set_major_locator(mdates.YearLocator())
        axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axes[i].set_title(f"Rupturas en {variable}")
        axes[i].grid(True)
        axes[i].legend()

    # Configuración general del gráfico
    plt.suptitle(f"Detección de rupturas estructurales, n={n_bkps}", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

# ================================================================================================================= #
 
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm

def detect_structural_breaks(y, X, model_type="l2", n_bkps=5):
    """
    Detect structural breaks in a multivariate system and validate them with statistical tests.

    This function combines automatic detection of structural breaks using the `ruptures` library
    with a statistical validation process. It identifies significant changes in the relationship
    between a dependent variable (`y`) and a set of independent variables (`X`) over time.

    The process involves:
    1. Automatic detection of breakpoints in the system using a cost-based segmentation algorithm.
    2. Validation of each detected breakpoint with an F-test comparing two models:
       - A baseline model without a structural break.
       - An alternative model including an indicator variable for the post-break period.

    Parameters:
    ----------
    y : pandas.Series
        The dependent variable, with a DateTimeIndex representing the time periods.
    X : pandas.DataFrame
        The independent variables, with a DateTimeIndex that matches `y`.
    model_type : str, optional
        The cost function to be used by the `ruptures` library for detecting breakpoints.
        Common options include:
        - "l2" (default): Least squares segmentation.
        - "linear": Linear regression cost.
        - "rbf": Radial basis function segmentation.
    n_bkps : int, optional
        The maximum number of breakpoints to detect in the data. Defaults to 5.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the following columns:
        - `break_date`: The date of each detected structural break.
        - `F-Statistic`: The F-statistic from the model comparison test.
        - `p-value`: The p-value associated with the F-test, indicating the significance
          of the structural break.

    Raises:
    ------
    ValueError
        If `y` or `X` does not have a DateTimeIndex.

    Examples:
    --------
    # Simulated data
    dates = pd.date_range("2020-01-01", "2022-12-31", freq="M")
    y = pd.Series(range(len(dates)), index=dates)  # Dependent variable
    X = pd.DataFrame({"costs": range(len(dates))}, index=dates)  # Independent variables

    # Detect structural breaks
    results = detect_structural_breaks(y, X, model_type="l2", n_bkps=3)
    print(results)

    Notes:
    -----
    - The function first aligns the indices of `y` and `X` to ensure consistency.
    - Detected breakpoints are validated using an F-test, which assumes normally distributed errors.
    - The F-test compares the fit of two Ordinary Least Squares (OLS) models:
      one with and one without a structural break.
    - This approach assumes the breakpoints are detected globally across all variables in `y` and `X`.

    """
    # Ensure y and X have matching DateTimeIndexes
    if not isinstance(y.index, pd.DatetimeIndex) or not isinstance(X.index, pd.DatetimeIndex):
        raise ValueError("Both y and X must have DateTimeIndex.")
    
    # Align y and X to ensure their indices match
    y, X = y.align(X, join="inner")
    
    # Combine y and X into a single matrix for ruptures
    data = pd.concat([y, X], axis=1).values
    
    # Detect breakpoints using ruptures
    model = rpt.Binseg(model=model_type).fit(data)
    breakpoints = model.predict(n_bkps=n_bkps)
    
    # Prepare results
    results = []
    for breakpoint in breakpoints[:-1]:  # Exclude the last index (end of data)
        break_date = y.index[breakpoint]
        
        # Create the post-break indicator
        post_break = (y.index >= break_date).astype(int)
        
        # Prepare the data for the model without the structural break
        X_no_break = sm.add_constant(X)
        model_no_break = sm.OLS(y, X_no_break).fit()
        
        # Prepare the data for the model with the structural break
        X_with_break = sm.add_constant(X)
        X_with_break["post_break"] = post_break
        model_with_break = sm.OLS(y, X_with_break).fit()
        
        # Perform the F-test to compare models
        f_stat, p_value, _ = model_with_break.compare_f_test(model_no_break)
        
        results.append({"break_date": break_date, "F-Statistic": f_stat, "p-value": p_value})
    
    return pd.DataFrame(results)

# ================================================================================================================= #

from statsmodels.stats.outliers_influence import variance_inflation_factor

def calcular_vif(dataframe):
    """
    Calcula el VIF (Factor de Inflación de la Varianza) para las columnas de un DataFrame.

    Parámetros:
    - dataframe: DataFrame de pandas con las variables independientes.

    Retorna:
    - DataFrame con dos columnas: Variable y VIF.
    """
    # Asegurarse de que no haya valores nulos en el dataframe
    if dataframe.isnull().sum().sum() > 0:
        raise ValueError("El DataFrame contiene valores nulos. Por favor, elimínalos o imputa los valores faltantes.")
    
    # Añadir una constante para representar el intercepto en el cálculo del VIF
    dataframe = dataframe.assign(constante=1)
    
    # Calcular el VIF para cada columna
    vif_data = pd.DataFrame({
        "Variable": dataframe.columns,
        "VIF": [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    })
    
    # Eliminar la constante del resultado final
    vif_data = vif_data[vif_data["Variable"] != "constante"]
    
    # Ordenar el resultado por VIF descendente
    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

# ================================================================================================================= #

import statsmodels.api as sm

def ols_backward_elimination(data, target, significance_level=0.05):
    """
    Realiza backward elimination para un modelo OLS basado en el p-valor de las variables.
    
    Args:
        data (pd.DataFrame): DataFrame con las variables independientes.
        target (pd.Series): Variable dependiente.
        significance_level (float): Nivel de significancia para eliminar variables.
    
    Returns:
        model: Modelo final ajustado.
        remaining_vars: Lista de variables seleccionadas.
    """
    variables = data.columns.tolist()
    while len(variables) > 0:
        # Ajustar el modelo OLS
        X = sm.add_constant(data[variables])  # Agregar la constante
        model = sm.OLS(target, X).fit(cov_type='HC1')
        
        # Obtener p-valores
        p_values = model.pvalues.iloc[1:]  # Excluir el p-valor de la constante
        max_p_value = p_values.max()  # Mayor p-valor
        
        if max_p_value > significance_level:
            # Eliminar la variable con el mayor p-valor
            excluded_var = p_values.idxmax()
            print(f"variable eliminada: {excluded_var}; p-valor: {max_p_value:.4f}")
            variables.remove(excluded_var)
        else:
            break
    
    # Modelo final
    final_model = sm.OLS(target, sm.add_constant(data[variables])).fit()
    return final_model, variables

# ================================================================================================================= #

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def plot_scatter_matrix(df, trendline=False):
    """
    Grafica todos los scatterplots de variables numéricas como subplots en un solo gráfico.
    Siempre muestra el R^2 en cada subplot y opcionalmente agrega una línea de tendencia.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        trendline (bool): Si es True, agrega una línea de tendencia en los gráficos.
    """
    # Filtrar variables numéricas
    numeric_cols = df.select_dtypes(include=np.number).columns
    num_vars = len(numeric_cols)
    
    if num_vars < 2:
        raise ValueError("El DataFrame debe tener al menos dos variables numéricas para crear scatterplots.")
    
    # Crear subplots
    fig, axes = plt.subplots(num_vars, num_vars, figsize=(15, 15), constrained_layout=True)
    fig.suptitle("Diagramas de dispersión", fontsize=16)
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogramas de la variable
                ax.hist(df[col1], bins=20, color='lightblue', alpha=0.7)
                ax.set_title(f'{col1}', fontsize=10)
            else:
                # Scatterplot entre dos variables
                sns.scatterplot(x=df[col2], y=df[col1], ax=ax, s=10, alpha=0.7)
                
                # Calcular línea de tendencia y R^2
                x_col = df[col2].values.reshape(-1, 1)
                y_col = df[col1].values
                model = LinearRegression().fit(x_col, y_col)
                y_pred = model.predict(x_col)
                r2 = r2_score(y_col, y_pred)
                
                # Mostrar R² con anotación
                ax.text(
                    0.05, 0.85, f'$R^2$ = {r2:.3f}',  # Siempre mostrar R²
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.5)
                )
                
                if trendline:
                    # Dibujar línea de tendencia si trendline=True
                    ax.plot(df[col2], y_pred, color='red', linewidth=1)
            
            # Configurar ejes
            if i == num_vars - 1:
                ax.set_xlabel(col2, fontsize=8)
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
                
            if j == 0:
                ax.set_ylabel(col1, fontsize=8)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)
    
    plt.show()

# ================================================================================================================= #

def plot_homoscedasticity(data):
    """
    Plot predicted values vs residuals for each variable in a DataFrame to check for homoscedasticity,
    and calculate the Breusch-Pagan test for each variable.

    Parameters:
    ----------
    data : pandas.DataFrame
        A DataFrame where each column represents a variable.

    Returns:
    -------
    None. Displays subplots of predicted values vs residuals along with p-value and a verdict for homoscedasticity.
    """
    num_vars = data.shape[1]
    cols = 2  # Number of columns for subplots
    rows = (num_vars + 1) // cols  # Determine the number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2 * rows))
    axes = axes.flatten()

    # Convert datetime index to numerical (years as float)
    time_as_float = data.index.year + (data.index.month - 1) / 12

    for i, col in enumerate(data.columns):
        # Fit a simple OLS model for each variable
        X = sm.add_constant(time_as_float)  # Time trend as predictor
        y = data[col]
        model = sm.OLS(y, X).fit()
        
        # Get predicted values and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Perform the Breusch-Pagan test
        exog = np.hstack([np.ones((len(y_pred), 1)), y_pred.reshape(-1, 1)])
        lm_stat, lm_pval, _, _ = het_breuschpagan(residuals, exog)
        verdict = "Homocedástica" if lm_pval > 0.05 else "Heterocedástica"

        # Plot predicted vs residuals
        axes[i].scatter(y_pred, residuals, alpha=0.6)
        axes[i].axhline(0, color='red', linestyle='--', linewidth=1)
        axes[i].set_title(f"{col} | p={lm_pval:.3f}: {verdict}", size=11)
        axes[i].set_xlabel("Predicciones")
        axes[i].set_ylabel("Residuos")

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Revisión de homocedasticidad, estadístico Breusch-Pagan", y=1.0)
    plt.tight_layout()
    plt.show()



def plot_correlation_heatmap(df, method='pearson'):
    
    # Acortar nombres de vriables para mejorar la visualización:
    truncated_labels = [col[:15] + "..." if len(col) > 10 else col for col in df.columns]
    
    # Colocar etiquetas centradas:
    posición_ticks = np.arange(len(df.columns)) + 0.5

    # Graficar mapa de calor de correlaciones no lineales:
    corr = df.corr(method=method) # 'pearson', 'spearman', 'kendall'
    mask = np.triu(np.ones_like(corr, dtype=bool))
    plt.figure(figsize=(6, 6))
    sns.heatmap(data=corr, mask=mask, cmap='coolwarm', center=0, linewidths=0.5, annot=True, annot_kws={"size": 6}, cbar=False)
    plt.title(f'Mapa de calor de correlaciones {method.capitalize()}', size=13)
    plt.xticks(ticks=posición_ticks, labels=truncated_labels, fontsize=8)
    plt.yticks(ticks=posición_ticks, labels=truncated_labels, fontsize=8)
    plt.show()