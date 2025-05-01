from modulos.graficas import Graficas
from modulos.sensor import Sensor
from modulos.metodos import Metodos
from modulos.funciones import Funciones
from modulos.horno import Horno
from modulos.ruido import Ruido
from modulos.errores import Errores
from modulos.tablas import *
import pandas as pd
import numpy as np
import csv

if __name__ == "__main__":

    PT1000 = Sensor(PT1000_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
    TMP235 = Sensor(TMP235Q1DICT, "TMP235", "Voltaje (mV)", "lineal")
    TYPE_E = Sensor(TYPE_E_DICT, "TYPE_E", "Voltaje (mV)", "polinomial")
    NTCLE100E3 = Sensor(NTCLE100E3_DICT, "NTCLE100E3", "Resistencia (Ohmios)", "exponencial")


    print("---------------------------------------------------")
    print("PUNTO 1: ENCONTRAR UN SUBRANGO DE TEMPERATURAS")
    print("---------------------------------------------------")

    rango = Funciones.superposicionRangos(PT1000, TYPE_E, TMP235, NTCLE100E3)
    print("\n *Rango de temperaturas", rango, '\n')
    Graficas.graficar_rangos_sensores([PT1000, TMP235, TYPE_E, NTCLE100E3], (0, 100))
    

    print("---------------------------------------------------")
    print("PUNTO 2: ENCONTRAR FUNCION CARACTERISTICA")
    print("---------------------------------------------------")


    Graficas.graficar_sensor_con_curva(PT1000)
    Graficas.graficar_sensor_con_curva(TMP235)
    Graficas.graficar_sensor_con_curva(TYPE_E)
    Graficas.graficar_sensor_con_curva(NTCLE100E3)
    

    print("\n *Parametros en rango completo: \n")
    print("Sensor    ", "Parametros")
    print("PT1000    :", PT1000.parametros)
    print("TYPE_TMP  :", TMP235.parametros)
    print("TYPE_E    :", TYPE_E.parametros)
    print("NTCLE100E3:", NTCLE100E3.parametros)

    sub_rango = (0, 100)
    PT1000_sub_DICT = Funciones.subRangoSensores(PT1000_DICT, sub_rango)
    TYPE_E_sub_DICT = Funciones.subRangoSensores(TYPE_E_DICT, sub_rango)
    TMP235_sub_DICT = Funciones.subRangoSensores(TMP235Q1DICT, sub_rango)
    NTCLE100E3_sub_DICT = Funciones.subRangoSensores(NTCLE100E3_DICT, sub_rango)


    PT1000 = Sensor(PT1000_sub_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
    TMP235 = Sensor(TMP235_sub_DICT, "TMP235", "Voltaje (mV)", "lineal")
    TYPE_E= Sensor(TYPE_E_sub_DICT, "TYPE_E", "Voltaje (mV)", "polinomial")
    NTCLE100E3 = Sensor(NTCLE100E3_sub_DICT, "NTCLE100E3", "Resistencia (Ohmios)", "exponencial")


    print("\n *Parametros calculados en un rango especifico: \n")

    print("PT1000     :", PT1000.parametros)
    print("TMP235     :", TMP235.parametros)
    print("TYPE_E     :", TYPE_E.parametros) 
    print("NTCLE100E3 :", NTCLE100E3.parametros)


    # Puntos de la tabla con su curva
    # Graficas.graficar_sensor_con_curva(PT1000)
    # Graficas.graficar_sensor_con_curva(TYPE_E)
    # Graficas.graficar_sensor_con_curva(TMP235)
    # Graficas.graficar_sensor_con_curva(NTCLE100E3)

    rmse_PT1000 = Metodos.rmse(PT1000.valores, PT1000.calcularValores(PT1000.temperaturas))
    rmse_TYPE_E = Metodos.rmse(TYPE_E.valores, TYPE_E.calcularValores(TYPE_E.temperaturas))
    rmse_TYPE_TMP = Metodos.rmse(TMP235.valores, TMP235.calcularValores(TMP235.temperaturas))
    rmse_TYPE_NTCLE = Metodos.rmse(NTCLE100E3.valores, NTCLE100E3.calcularValores(NTCLE100E3.temperaturas))

    print("\n *Mean Square Error")

    print("MSE PT1000", rmse_PT1000**2)
    print("MSE TYPE_E", rmse_TYPE_E**2)
    print("MSE TYPE_TMP", rmse_TYPE_TMP**2)
    print("MSE NTCLE100E3", rmse_TYPE_NTCLE**2)
    print("\n")


    print("---------------------------------------------------")
    print("PUNTO 3: SIMULACION DEL HORNO")
    print("---------------------------------------------------")

    X = 60
    Y = 15
    Z = 30
    W = 25
    T0 = 0

    # longitud = Y + W
    horno = Horno(X, Y, Z, W, T0)
    print("\n *Parametros del horno: \n")
    print("X:", X)
    print("Y:", Y)
    print("Z:", Z)
    print("W:", W)
    print("T0:", T0)
    print("\n")
    Graficas.graficar_dos_lineas(horno.temperaturas, horno.temperaturas, show=True, title="Temperatura del horno", ylabel="Temperatura (°C)", xlabel="Tiempo (s)", color='cornflowerblue', save=True, nombre="temperatura_horno.png")
    #Graficas.grafica_y(horno.temperaturas, show=True, estilo='o', color="orangered",title="Temperatura del horno", ylabel="Temperatura (°C)", xlabel="Tiempo (s)", save= True, nombre="temperatura_horno.png")


    print("---------------------------------------------------")
    print("PUNTO 4: SIMULACION DE SENSORES")
    print("---------------------------------------------------")

    def simular_sensor_sin_error(horno, sensor):
        dict_sensor = {}
        for temperatura in horno.temperaturas:
            valor = sensor.calcularValores(temperatura)
            dict_sensor[temperatura] = float(valor)
        return dict_sensor
    
    PT1000_ideal = simular_sensor_sin_error(horno, PT1000)
    TYPE_E_ideal = simular_sensor_sin_error(horno, TYPE_E)
    TMP235_ideal = simular_sensor_sin_error(horno, TMP235)
    NTCLE100E3_ideal = simular_sensor_sin_error(horno, NTCLE100E3)




    def simular_sensor_con_error(horno, sensor, rmse=None, p_outlier=0.005):
        dict_sensor = {}
        dict_errores = {}
        lista = []
        for temperatura in horno.temperaturas:
            valor = sensor.calcularValores(temperatura)

            # Agregar Outliers
            outlier_value = Ruido.generarOutlier(p_outlier)
            # Error sobre la medida
            error = Errores.calcularErrorMedida(sensor, temperatura, valor, rmse)

            error_gaussiano = np.random.normal(0, error, 1)[0]

            valor = valor + error_gaussiano + (valor * outlier_value)
            dict_errores[temperatura] = Errores.sumaErrores(error, rmse)
            dict_sensor[temperatura] = float(valor)
            lista.append(valor)
        return dict_sensor, dict_errores, lista
    


    PT1000_simulado, e_PT1000, lista_PT1000 = simular_sensor_con_error(horno, PT1000, rmse_PT1000 , 0.010)
    TYPE_E_simulado, e_TYPE_E, lista_TYPE_E = simular_sensor_con_error(horno, TYPE_E, rmse_TYPE_E,0.010)
    TMP235_simulado, e_TMP235, lista_TMP = simular_sensor_con_error(horno, TMP235, rmse_TYPE_TMP,0.010)
    NTCLE100E3_simulado, e_TYPE_NTCLE100E3, lista_NTCLE = simular_sensor_con_error(horno, NTCLE100E3, rmse_TYPE_NTCLE ,0.010)

    # guardar en un archivo csv
    df = pd.DataFrame({
        "PT1000": lista_PT1000,
        "TMP235": lista_TMP,
        "TYPE_E": lista_TYPE_E,
        "NTCLE100E3": lista_NTCLE,
        "Temperatura": horno.temperaturas
    })
    df.to_csv("datos/simulacion.csv", index=False)
        
    


    Graficas.graficar_dos_lineas(PT1000_ideal.values(), PT1000_simulado.values(), show=True, title="PT1000", ylabel="Resistencia (Ohmios)", xlabel="Tiempo (s)", color='salmon', save=True, nombre="PT1000_simulado.png", label1 = "Simulado", label2 = "Ideal")
    Graficas.graficar_dos_lineas(TYPE_E_ideal.values(), TYPE_E_simulado.values(), show=True, title="TYPE_E", ylabel="Voltaje (mV)", xlabel="Tiempo (s)", color='salmon', save=True, nombre="TYPE_E_simulado.png", label1 = "Simulado", label2 = "Ideal")
    Graficas.graficar_dos_lineas(TMP235_ideal.values(), TMP235_simulado.values(), show=True, title="TMP235", ylabel="Voltaje (mV)", xlabel="Tiempo (s)", color='salmon', save=True, nombre="TMP235_simulado.png", label1 = "Simulado", label2 = "Ideal")
    Graficas.graficar_dos_lineas(NTCLE100E3_ideal.values(), NTCLE100E3_simulado.values(), show=True, title="NTCLE100E3", ylabel="Resistencia (Ohmios)", xlabel="Tiempo (s)", color='salmon', save=True, nombre="NTCLE100E3_simulado.png", label1 = "Simulado", label2 = "Ideal")


    Graficas.grafica_y(e_PT1000.values(), show=True,  estilo='o', title="Errores PT1000", ylabel="Error (Ohmios)", xlabel="Tiempo (s)", color='cornflowerblue', save=True, nombre="error_PT1000.png")
    Graficas.grafica_y(e_TYPE_E.values(), show=True, estilo='o', title="Errores TYPE_E", ylabel="Error (mV)", xlabel="Tiempo (s)", color='cornflowerblue', save=True, nombre="error_TYPE_E.png")
    Graficas.grafica_y(e_TMP235.values(), show=True, estilo='o', title="Errores TMP235", ylabel="Error (mV)", xlabel="Tiempo (s)", color='cornflowerblue', save=True, nombre="error_TMP235.png")
    Graficas.grafica_y(e_TYPE_NTCLE100E3.values(), show=True, estilo='o', title="Errores NTCLE100E3", ylabel="Error (Ohmios)", xlabel="Tiempo (s)", color='cornflowerblue', save=True, nombre="error_NTCLE100E3.png")


    print("\n *Simulacion con ruidos diferentes: \n")
    
    def simular_sensores_ruido_variados(horno,  p_outlier=0.005):
        dict_PT1000 = {}
        dict_TYPE_E = {}
        dict_TMP = {}
        dict_NTCLE100E3 = {}

        for temperatura in horno.temperaturas:
            valor_PT1000 = PT1000.calcularValores(temperatura)
            valor_TYPE_E = TYPE_E.calcularValores(temperatura)
            valor_TMP = TMP235.calcularValores(temperatura)
            valor_NTCLE100E3 = NTCLE100E3.calcularValores(temperatura)

            # Agregar Outliers
            outlier_value_PT1000 = Ruido.generarOutlier(p_outlier)
            outlier_value_TYPE_E = Ruido.generarOutlier(p_outlier)
            outlier_value_TMP = Ruido.generarOutlier(p_outlier)
            outlier_value_NTCLE100E3 = Ruido.generarOutlier(p_outlier)

            # Error sobre la medida
            error_PT1000 = Errores.calcularErrorMedida(PT1000, temperatura, valor_PT1000, rmse_PT1000)
            error_TYPE_E = Errores.calcularErrorMedida(TYPE_E, temperatura, valor_TYPE_E, rmse_TYPE_E)
            error_TMP = Errores.calcularErrorMedida(TMP235, temperatura, valor_TMP, rmse_TYPE_TMP)
            error_NTCLE100E3 = Errores.calcularErrorMedida(NTCLE100E3, temperatura, valor_NTCLE100E3, rmse_TYPE_NTCLE)

 
            ruido_PT1000 = Ruido.ruido_gaussiano(0.0, error_PT1000)
            ruido_TYPE_E = Ruido.ruido_uniforme(-error_TYPE_E, error_TYPE_E)
            ruido_TMP = Ruido.ruido_poisson(error_TMP)
            ruido_NTCLE100E3 = Ruido.ruido_laplace(0.0, error_NTCLE100E3)




            dict_PT1000[temperatura] = valor_PT1000 + ruido_PT1000 + (valor_PT1000 * outlier_value_PT1000)
            dict_TYPE_E[temperatura] = valor_TYPE_E + ruido_TYPE_E + (valor_TYPE_E * outlier_value_TYPE_E)
            dict_TMP[temperatura] = valor_TMP + ruido_TMP + (valor_TMP * outlier_value_TMP)
            dict_NTCLE100E3[temperatura] = valor_NTCLE100E3 + ruido_NTCLE100E3 + (valor_NTCLE100E3 * outlier_value_NTCLE100E3)
        return dict_PT1000, dict_TYPE_E, dict_TMP, dict_NTCLE100E3

    PT_1000_simulado, TYPE_E_simulado, TMP235_simulado, NTCLE100E3_simulado = simular_sensores_ruido_variados(horno, 0.010)

    Graficas.graficar_dos_lineas(PT1000_ideal.values(), PT1000_simulado.values(), show=True, title="PT1000 (Ruido gaussiano)", ylabel="Resistencia (Ohmios)", xlabel="Tiempo (s)", color="orangered", save=True, nombre="PT1000_ruido_gaussiano.png", label1 = "Simulado", label2 = "Ideal")
    Graficas.graficar_dos_lineas(TYPE_E_ideal.values(), TYPE_E_simulado.values(), show=True, title="TYPE_E (Ruido uniforme)", ylabel="Voltaje (mV)", xlabel="Tiempo (s)", color="orangered", save=True, nombre="TYPE_E_ruido_uniforme.png", label1 = "Simulado", label2 = "Ideal")
    Graficas.graficar_dos_lineas(TMP235_ideal.values(), TMP235_simulado.values(), show=True, title="TMP235 (Ruido poisson)", ylabel="Voltaje (mV)", xlabel="Tiempo (s)", color="orangered", save=True, nombre="TMP235_ruido_poisson.png", label1 = "Simulado", label2 = "Ideal")
    Graficas.graficar_dos_lineas(NTCLE100E3_ideal.values(), NTCLE100E3_simulado.values(), show=True, title="NTCLE100E3 (Ruido laplace)", ylabel="Resistencia (Ohmios)", color="orangered", xlabel="Tiempo (s)", save=True, nombre="NTCLE100E3_ruido_laplace.png", label1 = "Simulado", label2 = "Ideal")
