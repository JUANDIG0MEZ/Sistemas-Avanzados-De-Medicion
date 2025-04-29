from modulos.graficas import Graficas
from modulos.sensor import Sensor
from modulos.metodos import Metodos
from modulos.funciones import Funciones
from modulos.horno import Horno
from modulos.ruido import Ruido
from modulos.errores import Errores
import numpy as np
from tablas import *
import matplotlib.pyplot as plt


if __name__ == "__main__":

    PT1000 = Sensor(PT1000_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
    TYPE_K = Sensor(TYPE_K_DICT, "TYPE_K", "Voltaje (mV)", "polinomial")
    TYPE_E = Sensor(TYPE_E_DICT, "TYPE_E", "Voltaje (mV)", "polinomial")
    TYPE_TMP = Sensor(TMP235Q1DICT, "TMP235-Q1", "Voltaje (mV)", "lineal")
    NTCLE100E3 = Sensor(NTCLE100E3_DICT, "NTCLE100E3", "Resistencia (Ohmios)", "exponencial")

    rango = Funciones.superposicionRangos(PT1000, TYPE_K, TYPE_E, TYPE_TMP, NTCLE100E3)
    print("Superposicion de los rangos de los sensores", rango)
    #Graficas.graficar_rangos_sensores([PT1000, TYPE_K, TYPE_E, TYPE_TMP, NTCLE100E3338], (0, 100))

    # Graficas.graficar_sensor_con_curva(PT1000)
    # Graficas.graficar_sensor_con_curva(TYPE_K)
    # Graficas.graficar_sensor_con_curva(TYPE_E)
    # Graficas.graficar_sensor_con_curva(TYPE_TMP)
    # Graficas.graficar_sensor_con_curva(NTCLE100E3338)


    sub_rango = 0.6
    PT1000_sub_DICT = Funciones.subRango(PT1000_DICT, sub_rango)
    TYPE_K_sub_DICT = Funciones.subRango(TYPE_K_DICT, sub_rango)
    TYPE_E_sub_DICT = Funciones.subRango(TYPE_E_DICT, sub_rango)
    TYPE_TMP_sub_DICT = Funciones.subRango(TMP235Q1DICT, sub_rango)
    NTCLE100E3_sub_DICT = Funciones.subRango(NTCLE100E3_DICT, sub_rango)


    PT1000 = Sensor(PT1000_sub_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
    TYPE_K = Sensor(TYPE_K_sub_DICT, "TYPE_K", "Voltaje (mV)", "polinomial")
    TYPE_E= Sensor(TYPE_E_sub_DICT, "TYPE_E", "Voltaje (mV)", "polinomial")
    TYPE_TMP = Sensor(TYPE_TMP_sub_DICT, "TMP235-Q1", "Voltaje (mV)", "lineal")
    NTCLE100E3338 = Sensor(NTCLE100E3_sub_DICT, "NTCLE100E3", "Resistencia (Ohmios)", "exponencial")


    # Puntos de la tabla con su curva
    # Graficas.graficar_sensor_con_curva(PT1000)
    # Graficas.graficar_sensor_Errores de ajustecon_curva(TYPE_TMP)
    # Graficas.graficar_sensor_con_curva(NTCLE100E3338)


    print("--------------------------")
    print("Parametros de los sensores")
    print("--------------------------")
    print("Parametros del sensor PT1000")
    print(PT1000.obtenerParametros())
    print("Parametros sensor TYPE_K")
    print(TYPE_K.obtenerParametros())
    print("Parametros sensor TYPE_E")
    print(TYPE_E.obtenerParametros())
    print("Parametros sensor TYPE_TMP")
    print(TYPE_TMP.obtenerParametros())
    print("Parametros sensor NTCLE100E3338")
    print(NTCLE100E3338.obtenerParametros())


    print("--------------------------")
    print("Crear Horno")
    X = 100
    Y = 30
    Z = 30
    W = 50
    T0 = 0

    # longitud = Y + W
    horno = Horno(X, Y, Z, W, T0)


    

    # Graficas.grafica_y(PT_1000_simulado.values())
    # Graficas.grafica_y(TYPE_K_simulado.values())
    # Graficas.grafica_y(TYPE_E_simulado.values())
    # Graficas.grafica_y(TYPE_TMP_simulado.values())
    # Graficas.grafica_y(NTCLE100E3338_simulado.values())


    print("--------------------------")
    print("Agregar Errores")

    print("--------------------------")
    print("Error de ajuste de la curva")
    print("--------------------------")

    rmse_PT1000 = Metodos.rmse(PT1000.valores, PT1000.calcularValores(PT1000.temperaturas))
    rmse_TYPE_K = Metodos.rmse(TYPE_K.valores, TYPE_K.calcularValores(TYPE_K.temperaturas))
    rmse_TYPE_E = Metodos.rmse(TYPE_E.valores, TYPE_E.calcularValores(TYPE_E.temperaturas))
    rmse_TYPE_TMP = Metodos.rmse(TYPE_TMP.valores, TYPE_TMP.calcularValores(TYPE_TMP.temperaturas))
    rmse_TYPE_NTCLE = Metodos.rmse(NTCLE100E3338.valores, NTCLE100E3338.calcularValores(NTCLE100E3338.temperaturas))


    def simular_sensor(horno, sensor, p_outlier=0.005):
        dict_sensor = {}
        for temperatura in horno.temperaturas:
            valor = sensor.calcularValores(temperatura)
            # Agregar Outliers
            outlier_value = Ruido.generarOutlier(p_outlier)




            valor = valor + valor * outlier_value
            dict_sensor[temperatura] = float(valor)
        return dict_sensor
    


    PT_1000_simulado = simular_sensor(horno, PT1000, outlier=0.010)
    TYPE_K_simulado = simular_sensor(horno, TYPE_K, outlier=0.010)
    TYPE_E_simulado = simular_sensor(horno, TYPE_E, outlier=0.010)
    TYPE_TMP_simulado = simular_sensor(horno, TYPE_TMP, outlier=0.010)
    NTCLE100E3338_simulado = simular_sensor(horno, NTCLE100E3338,  outlier=0.010)

    error_PT1000 = 0.1
    error_TYPE_K = 2.2
    error_TYPE_E = 1.7
    error_TYPE_TMP = 2.5
    error_NTCLE100E3338 = 0.0
    ruido_homocedastico_PT1000 = Ruido("gaussiano", 0.0, error_PT1000, len_temperaturas).valores
    ruido_homocedastico_TYPE_K = Ruido("gaussiano", 0.0, error_TYPE_K, len_temperaturas).valores
    ruido_homocedastico_TYPE_E = Ruido("gaussiano", 0.0, error_TYPE_E, len_temperaturas).valores
    ruido_homocedastico_TYPE_TMP = Ruido("gaussiano", 0.0, error_TYPE_TMP, len_temperaturas).valores
    ruido_homocedastico_NTCLE100E3338 = Ruido("gaussiano", 0.0, error_NTCLE100E3338, len_temperaturas).valores

    print("--------------------------")
    print("Ruido heterocedastico")



    error_PT1000 = 0.0 / 10
    error_TYPE_K = 0.75 / 10
    error_TYPE_E = 0.5 /10
    error_TYPE_TMP = 0.0 / 10
    error_NTCLE100E3338 = 5 /10
    ruido_heterocedastico_PT1000 = Ruido("gaussiano", 0.0, error_PT1000, len_temperaturas).valores
    ruido_heterocedastico_TYPE_K = Ruido("gaussiano", 0.0, error_TYPE_K, len_temperaturas).valores
    ruido_heterocedastico_TYPE_E = Ruido("gaussiano", 0.0, error_TYPE_E, len_temperaturas).valores
    ruido_heterocedastico_TYPE_TMP = Ruido("gaussiano", 0.0, error_TYPE_TMP, len_temperaturas).valores
    ruido_heterocedastico_NTCLE100E3338 = Ruido("gaussiano", 0.0, error_NTCLE100E3338, len_temperaturas).valores


    valores_PT_1000 = np.array([i for i in PT_1000_simulado.values()])
    valores_TYPE_K = np.array([i for i in TYPE_K_simulado.values()])
    valores_TYPE_E = np.array([i for i in TYPE_E_simulado.values()])
    valores_TYPE_TMP = np.array([i for i in TYPE_TMP_simulado.values()])
    valores_NTCLE100E3338 = np.array([i for i in NTCLE100E3338_simulado.values()])

    print("VALORES PT1000", valores_PT_1000)
    print("Error heterocestatico PT1000", ruido_heterocedastico_PT1000)

    valores_PT_1000_con_error = valores_PT_1000 + (valores_PT_1000 * ruido_heterocedastico_PT1000) + ruido_PT1000 + ruido_homocedastico_PT1000
    valores_TYPE_K_con_error = valores_TYPE_K + (valores_TYPE_K * ruido_heterocedastico_TYPE_K) + ruido_TYPE_K + ruido_homocedastico_TYPE_K
    valores_TYPE_E_con_error = valores_TYPE_E + (valores_TYPE_E * ruido_heterocedastico_TYPE_E) + ruido_TYPE_E + ruido_homocedastico_TYPE_E
    valores_TYPE_TMP_con_error = valores_TYPE_TMP + (valores_TYPE_TMP * ruido_heterocedastico_TYPE_TMP) + ruido_TYPE_TMP + ruido_homocedastico_TYPE_TMP
    valores_NTCLE100E3338_con_error = valores_NTCLE100E3338 + (valores_NTCLE100E3338 * ruido_heterocedastico_NTCLE100E3338) + ruido_NTCLE100E3338 + ruido_homocedastico_NTCLE100E3338


    # Graficas.grafica_y(valores_PT_1000_con_error, show=False)
    # Graficas.grafica_y(valores_PT_1000, estilo='-')

    # Graficas.grafica_y(valores_TYPE_K_con_error, show=False)
    # Graficas.grafica_y(valores_TYPE_K, estilo='-')

    # Graficas.grafica_y(valores_TYPE_E_con_error, show=False)
    # Graficas.grafica_y(valores_TYPE_E, estilo='-')






    print("--------------------------")
    print("Crear los ruidos variados")

    len_temperaturas = len(horno.temperaturas)
    ruido_PT1000 = Ruido("gaussiano", 0.0, rmse_PT1000, len_temperaturas).valores
    ruido_TYPE_K = Ruido("uniforme", -rmse_TYPE_K, rmse_TYPE_K, len_temperaturas).valores
    ruido_TYPE_E = Ruido("cauchy", 0.0, rmse_TYPE_E, len_temperaturas).valores
    ruido_TYPE_TMP = Ruido("poisson", 0.0, rmse_TYPE_TMP, len_temperaturas).valores
    ruido_NTCLE100E3338 = Ruido("laplace", 0.0, rmse_TYPE_NTCLE, len_temperaturas).valores


    print("--------------------------")
    print("Ruido homocedastico")

    error_PT1000 = 0.1
    error_TYPE_K = 2.2
    error_TYPE_E = 1.7
    error_TYPE_TMP = 2.5
    error_NTCLE100E3338 = 0.0
    ruido_homocedastico_PT1000 = Ruido("gaussiano", 0.0, error_PT1000, len_temperaturas).valores
    ruido_homocedastico_TYPE_K = Ruido("uniforme", -rmse_TYPE_K, error_TYPE_K, len_temperaturas).valores
    ruido_homocedastico_TYPE_E = Ruido("cauchy", 0.0, error_TYPE_E, len_temperaturas).valores
    ruido_homocedastico_TYPE_TMP = Ruido("poisson", 0.0, error_TYPE_TMP, len_temperaturas).valores
    ruido_homocedastico_NTCLE100E3338 = Ruido("laplace", 0.0, error_NTCLE100E3338, len_temperaturas).valores

    print("--------------------------")
    print("Ruido heterocedastico")



    error_PT1000 = 0.0 / 10
    error_TYPE_K = 0.75 / 10
    error_TYPE_E = 0.5 /10
    error_TYPE_TMP = 0.0 / 10
    error_NTCLE100E3338 = 5 /10
    ruido_heterocedastico_PT1000 = Ruido("gaussiano", 0.0, error_PT1000, len_temperaturas).valores
    ruido_heterocedastico_TYPE_K = Ruido("uniforme", -rmse_TYPE_K, error_TYPE_K, len_temperaturas).valores
    ruido_heterocedastico_TYPE_E = Ruido("cauchy", 0.0, error_TYPE_E, len_temperaturas).valores
    ruido_heterocedastico_TYPE_TMP = Ruido("poisson", 0.0, error_TYPE_TMP, len_temperaturas).valores
    ruido_heterocedastico_NTCLE100E3338 = Ruido("laplace", 0.0, error_NTCLE100E3338, len_temperaturas).valores


    valores_PT_1000 = np.array([i for i in PT_1000_simulado.values()])
    valores_TYPE_K = np.array([i for i in TYPE_K_simulado.values()])
    valores_TYPE_E = np.array([i for i in TYPE_E_simulado.values()])
    valores_TYPE_TMP = np.array([i for i in TYPE_TMP_simulado.values()])
    valores_NTCLE100E3338 = np.array([i for i in NTCLE100E3338_simulado.values()])

    print("VALORES PT1000", valores_PT_1000)
    print("Error heterocestatico PT1000", ruido_heterocedastico_PT1000)

    valores_PT_1000_con_error = valores_PT_1000 + (valores_PT_1000 * ruido_heterocedastico_PT1000) + ruido_PT1000 + ruido_homocedastico_PT1000
    valores_TYPE_K_con_error = valores_TYPE_K + (valores_TYPE_K * ruido_heterocedastico_TYPE_K) + ruido_TYPE_K + ruido_homocedastico_TYPE_K
    valores_TYPE_E_con_error = valores_TYPE_E + (valores_TYPE_E * ruido_heterocedastico_TYPE_E) + ruido_TYPE_E + ruido_homocedastico_TYPE_E
    valores_TYPE_TMP_con_error = valores_TYPE_TMP + (valores_TYPE_TMP * ruido_heterocedastico_TYPE_TMP) + ruido_TYPE_TMP + ruido_homocedastico_TYPE_TMP
    valores_NTCLE100E3338_con_error = valores_NTCLE100E3338 + (valores_NTCLE100E3338 * ruido_heterocedastico_NTCLE100E3338) + ruido_NTCLE100E3338 + ruido_homocedastico_NTCLE100E3338


    Graficas.grafica_y(valores_PT_1000_con_error, show=False)
    Graficas.grafica_y(valores_PT_1000, estilo='-')

    Graficas.grafica_y(valores_TYPE_K_con_error, show=False)
    Graficas.grafica_y(valores_TYPE_K, estilo='-')

    Graficas.grafica_y(valores_TYPE_E_con_error, show=False)
    Graficas.grafica_y(valores_TYPE_E, estilo='-')

    Graficas.grafica_y(valores_TYPE_TMP_con_error, show=False)
    Graficas.grafica_y(valores_TYPE_TMP, estilo='-')

    Graficas.grafica_y(valores_NTCLE100E3338_con_error, show=False)
    Graficas.grafica_y(valores_NTCLE100E3338, estilo='-')

    