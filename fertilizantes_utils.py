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