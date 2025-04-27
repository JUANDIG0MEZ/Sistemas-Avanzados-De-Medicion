from modulos.graficas import Graficas
from modulos.sensor import Sensor
from modulos.metodos import Metodos
import numpy as np
from tablas import *
import matplotlib.pyplot as plt

class Horno:
    def __init__(self, X, Y, Z, W, T0=0):
        """
        Este horno sigue un perfil de temperatura conocido. La temperatura
        incrementa X grados en Y segundo, para luego decrecer Z grados en W segundos"""
        self.X = X
        self.Y = Y
        self.Z = Z
        self.W = W
        self.T0 = T0
        self.temperaturas = self.generar_temperaturas()
    
    def generar_temperaturas(self):
        """
        Esta funcion genera las temperaturas del horno en un tiempo determinado
        """
        tiempo = np.arange(0, self.Y + self.W, 1)
        temperaturas = []
        for t in tiempo:
            if t < self.Y:
                temperatura = self.T0 + (self.X / self.Y) * t
            else:
                temperatura = self.T0 + self.X - (self.Z / self.W) * (t - self.Y)
            temperaturas.append(temperatura)
        return np.array(temperaturas) + 273.15

    def graficar_perfil_temperatura(self):
        """
        Esta funcion grafica el perfil de temperatura del horno
        """
        plt.plot(self.temperaturas)
        plt.show()


if __name__ == "__main__":

    sensor_PT1000 = Sensor(PT1000_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
    sensor_TYPE_K = Sensor(TYPE_K_DICT, "Type K", "Voltaje (mV)", "polinomial")
    sensor_TYPE_E = Sensor(TYPE_E_DICT, "Type E", "Voltaje (mV)", "polinomial")
    sensor_TYPE_TMP = Sensor(TMP235Q1DICT, "TMP235-Q1", "Voltaje (mV)", "lineal")
    sensor_NTCLE100E3338 = Sensor(NTCLE100E3338_DICT, "NTCLE100E3338", "Resistencia (Ohmios)", "exponencial")


    Graficas.graficar_rangos_sensores([sensor_PT1000, sensor_TYPE_K, sensor_TYPE_E, sensor_TYPE_TMP, sensor_NTCLE100E3338], (0, 100))

    Graficas.graficar_sensor_con_curva(sensor_PT1000)
    Graficas.graficar_sensor_con_curva(sensor_TYPE_K)
    Graficas.graficar_sensor_con_curva(sensor_TYPE_E)
    Graficas.graficar_sensor_con_curva(sensor_TYPE_TMP)
    Graficas.graficar_sensor_con_curva(sensor_NTCLE100E3338)





    # PT1000_60_DICT = extraer_submuestra(PT1000_DICT, 0.6)
    # TYPE_K_60_DICT = extraer_submuestra(TYPE_K_DICT, 0.6)
    # TYPE_E_60_DICT = extraer_submuestra(TYPE_E_DICT, 0.6)
    # TYPE_TMP_60_DICT = extraer_submuestra(TMP235Q1DICT, 0.6)
    # NTCLE100E3338_DICT = extraer_submuestra(NTCLE100E3338_DICT, 0.6)

    # sensor_PT1000_60 = Sensor(PT1000_60_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
    # sensor_TYPE_K_60 = Sensor(TYPE_K_60_DICT, "Type K", "Voltaje (mV)", "polinomial")
    # sensor_TYPE_E_60 = Sensor(TYPE_E_60_DICT, "Type E", "Voltaje (mV)", "polinomial")
    # sensor_TYPE_TMP_60 = Sensor(TYPE_TMP_60_DICT, "TMP235-Q1", "Voltaje (mV)", "lineal")
    # sensor_NTCLE100E3338_60 = Sensor(NTCLE100E3338_DICT, "NTCLE100E3338", "Resistencia (Ohmios)", "exponencial")

    # lista_sensores = {
    #     "PT1000": sensor_PT1000_60,
    #     "TYPE_K": sensor_TYPE_K_60,
    #     "TYPE_E": sensor_TYPE_E_60,
    #     #"TYPE_TMP": sensor_TYPE_TMP_60,
    #     "NTCLE100E3338": sensor_NTCLE100E3338_60
    # }

    # Puntos de la tabla con su curva
    # Graficas.graficar_sensor_con_curva(sensor_PT1000_60)
    # Graficas.graficar_sensor_con_curva(sensor_TYPE_K_60)
    # Graficas.graficar_sensor_con_curva(sensor_TYPE_E_60)
    # Graficas.graficar_sensor_con_curva(sensor_TYPE_TMP_60)
    # Graficas.graficar_sensor_con_curva(sensor_NTCLE100E3338_60)


    # print("--------------------------")
    # print("Parametros de los sensores")
    # print("--------------------------")
    # print("Parametros del sensor PT1000")
    # print(sensor_PT1000_60.obtenerParametros())
    # print("Parametros sensor TYPE_K")
    # print(sensor_TYPE_K_60.obtenerParametros())
    # print("Parametros sensor TYPE_E")
    # print(sensor_TYPE_E_60.obtenerParametros())
    # print("Parametros sensor TYPE_TMP")
    # print(sensor_TYPE_TMP_60.obtenerParametros())
    # print("Parametros sensor NTCLE100E3338")
    # print(sensor_NTCLE100E3338_60.obtenerParametros())



    # print("--------------------------")
    # print("Errores de ajuste")
    # print("--------------------------")
    # print("PT1000:    ", Metodos.rmse(sensor_PT1000_60.valores, sensor_PT1000_60.calcularValores(sensor_PT1000_60.temperaturas)))
    # print("TYPE_K:    ", Metodos.rmse(sensor_TYPE_K_60.valores, sensor_TYPE_K_60.calcularValores(sensor_TYPE_K_60.temperaturas)))
    # print("TYPE_E:    ", Metodos.rmse(sensor_TYPE_E_60.valores, sensor_TYPE_E_60.calcularValores(sensor_TYPE_E_60.temperaturas)))
    # print("TYPE_TMP:  ", Metodos.rmse(sensor_TYPE_TMP_60.valores, sensor_TYPE_TMP_60.calcularValores(sensor_TYPE_TMP_60.temperaturas)))
    # print("NTCLE100E3338: ", Metodos.rmse(sensor_NTCLE100E3338_60.valores, sensor_NTCLE100E3338_60.calcularValores(sensor_NTCLE100E3338_60.temperaturas)))


    # print("--------------------------")
    # print("Crear Horno")
    # X = 100
    # Y = 60
    # Z = 30
    # W = 50
    # T0 = 0

    # longitud = Y + W
    # # horno = Horno(X, Y, Z, W, T0)

    # #horno.graficar_perfil_temperatura()


    # # print("--------------------------")
    # # print("Simulacion")
    # # print("--------------------------")

    # # simulacion = Simulacion(horno, num_iteraciones=5, lista_sensores=lista_sensores)

    # # print("--------------------------")
    # # print("Simulacion con varios gaussianos")
    # # print("--------------------------")

    # # sensoresSimulados = simulacion.simulacionGaussianos()


    # # Graficas.grafica_basica(sensoresSimulados["PT1000"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimulados["TYPE_K"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimulados["TYPE_E"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimulados["TYPE_TMP"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimulados["NTCLE100E3338"]["temperaturas_ruido"] - 273.15)
    # # Graficas.grafica_basica(horno.temperaturas )
    # # plt.show()



    # # print("--------------------------")
    # # print("Simulacion con varios ruidos")
    # # print("--------------------------")

    # # sensoresSimuladosVarios = simulacion.simulacionVariosRuidos()

    # # Graficas.grafica_basica(sensoresSimuladosVarios["PT1000"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimuladosVarios["TYPE_K"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimuladosVarios["TYPE_E"]["temperaturas_ruido"] )
    # # #Graficas.grafica_basica(sensoresSimuladosVarios["TYPE_TMP"]["temperaturas_ruido"] )
    # # Graficas.grafica_basica(sensoresSimuladosVarios["NTCLE100E3338"]["temperaturas_ruido"] - 273.15)
    # # Graficas.grafica_basica(horno.temperaturas )
    # # plt.show()



    # # print("--------------------------")
    # # print("Multples simulacion con Monte Carlo")
    # # print("--------------------------")

    # # montecarlo = simulacion.monteCarlo("gaussiano")



    # # print()

    # # Llamar a la función para graficar los histogramas
    # #graficar_histogramas_montecarlo(montecarlo, lista_sensores)
    # # valores_PT1000_ruido = Ruido.ruidoGaussiano(sensor_PT1000, 1.5)
    # # temperaturas_PT1000_ruido = Sensor.calcularTemperatura(sensor_PT1000, valores_PT1000_ruido)

    # # Graficas.graficar_sensor_con_curva(sensor_PT1000)
    # # Graficas.grafica_simple(temperaturas_PT1000_ruido, valores_PT1000_ruido)
    # # print("valores del PT_1000_ con ruido", valores_PT1000_ruido)
    # # plt.show()
