"""
Codigo mas organizado para el examen 1
"""

import matplotlib.pyplot as plt
import numpy as np
from tablas import *


class Metodos:
    @staticmethod
    def svd_obtenerParametros(A, b):
        """
        Esta funcion obtiene los parametros de la curva utilizando la descomposicion SVD
        """
        U, S, VT = np.linalg.svd(A, full_matrices=False)
        S_inv = np.diag(1 / S)
        A_inv = VT.T @ S_inv @ U.T
        return A_inv @ b

    @staticmethod
    def rmse(valores_reales, valores_calculados):
        """
        Esta funcion calcula los errores entre los valores y los valores calculados
        """
        errores = valores_reales - valores_calculados
        rmse = np.sqrt(np.mean(np.square(errores)))
        return rmse

    # @staticmethod
    # def calcularIncertidumbre(valores, )

class Sensor():
    def __init__(self, diccionario, nombre_sensor, unidades_valores, tipo_curva="lineal"):
        self.temperaturas, self.valores = self.obtener_datos(diccionario)
        self.length = len(self.valores)
        self.nombre_sensor = nombre_sensor
        self.unidades_valores = unidades_valores
        self.tipo_curva = tipo_curva
        self.parametros = None

    def obtener_datos(self, diccionario):
        """
        Esta funcion extrae los datos del sensor que se encuentran en el diccionario
        """
        T = []
        valores = []
        for temperatura, valor in diccionario.items():
            T.append(temperatura)
            valores.append(valor)

        # Convertimos las listas a arrays de numpy
        T = np.array(T) + 273.15
        valores = np.array(valores)
        return T, valores

    def calcularParametros(self):
        """
        Esta funcion calcula los parametros de la curva
        """
        b = np.array(self.valores).T
        if self.tipo_curva == "lineal":
            """
            R(T) = m*T + b ==> [T, 1] * [[m], [b]]
            """
            A = np.array([self.temperaturas, np.ones(self.length)]).T

        elif self.tipo_curva == "exponencial":
            """
            R(T) = A * e^(B/T) ==> ln(R(T)) = ln(A) + B/T ==> ln(R(T)) = [1/T, 1] * [[B], [ln(A)]]
            """
            A = np.array([1 / self.temperaturas, np.ones(len(self.valores))]).T
            b = np.log(b)

        elif self.tipo_curva == "polinomial":
            """
            V(T) = A + B * T + C * T^2 + D * T^3
            """
            A = np.array([np.ones(self.length), self.temperaturas, self.temperaturas**2, self.temperaturas ** 3]).T

        else: 
            raise ValueError("Tipo de curva no soportada")
        
        parametros = Metodos.svd_obtenerParametros(A, b)
        return parametros
    
    def obtenerParametros(self):
        """
        Esta funcion obtiene los parametros de la curva
        """
        if self.parametros is None:
            self.parametros = self.calcularParametros()
        return self.parametros

    def calcularValores(self, temperaturas):
        if self.parametros is None:
            self.parametros = self.calcularParametros()
        
        if self.tipo_curva == "lineal":
            """
            V(T) = m*T + b
            """
            m = self.parametros[0]
            b = self.parametros[1]
            return m * temperaturas + b
        elif self.tipo_curva == "exponencial":
            """
            R(T) = A * e^(B/T)
            """
            A = np.exp(self.parametros[1])
            B = self.parametros[0]
            return A * np.exp(B / (temperaturas + 273.15))
        elif self.tipo_curva == "polinomial":
            """
            V(T0) = A + B * T + C * T^2 + D * T^3
            """
            A = self.parametros[0]
            B = self.parametros[1]
            C = self.parametros[2]
            D = self.parametros[3]
            return A + B * temperaturas + C * temperaturas**2 + D * temperaturas**3
        else:
            raise ValueError("Tipo de curva no soportada")

    @staticmethod
    def calcularTemperatura(sensor, valores):
        """
        Esta funcion calcula la temperatura a partir de los valores
        """
        if sensor.parametros is None:
            sensor.parametros = sensor.calcularParametros()

        if sensor.tipo_curva == "lineal":
            """
            V(T) = m*T + b ==> T = (V(T) - b) / m
            """
            m = sensor.parametros[0]
            b = sensor.parametros[1]
            return (valores - b) / m

        elif sensor.tipo_curva == "exponencial":
            """
            R(T) = A * e^(B/T) ==> T = B / ln(R(T)/A)
            """
            A = np.exp(sensor.parametros[1])
            B = sensor.parametros[0]
            return B / np.log(valores / A)

        elif sensor.tipo_curva == "polinomial":
            """
            V(T) = A + B * T + C * T^2 + D * T^3
            """
            A = sensor.parametros[0]
            B = sensor.parametros[1]
            C = sensor.parametros[2]
            D = sensor.parametros[3]

            temperaturas = []
            for valor in valores:
                coef = [D, C, B, A - valor]
                # raices
                raices = np.roots(coef)
                # Filtramos las raices reales
                reales = raices[np.isreal(raices)].real
                # Elegimos la raiz valida y positiva
                posibles = reales[reales > 0]
                if len(posibles) == 0:
                    temperaturas.append(np.nan)
                else:
                    temperaturas.append(posibles.min())
            return np.array(temperaturas)

        else: 
            raise ValueError("Tipo de curva no soportada")

    def calcularIncertidumbre(self):
        """
        Esta funcion calcula la incertidumbre del sensor
        """
        pass


class Graficas():
    @staticmethod
    def graficar_sensor(sensor):
        """
        Esta funcion grafica los datos del sensor
        """
        plt.plot(sensor.temperaturas, sensor.valores, "o")
        plt.xlabel("Temperatura (°C)")   
        plt.ylabel(f"{sensor.unidades_valores}")
        plt.title(f"Grafica de {sensor.nombre_sensor}")
        plt.show()
    
    @staticmethod
    def graficar_sensor_con_curva(sensor):
        """
        Esta funcion grafica los datos del sensor junto con la curva ajustada
        """
        T = sensor.temperaturas
        if sensor.parametros is None: 
            sensor.parametros = sensor.calcularParametros()
        

        if sensor.tipo_curva == "lineal":
            y = sensor.parametros[0] * T + sensor.parametros[1]
        elif sensor.tipo_curva == "exponencial":
            y = np.exp(sensor.parametros[1] ) * np.exp(sensor.parametros[0] / T)
        elif sensor.tipo_curva == "polinomial":
            y = sensor.parametros[0] + sensor.parametros[1] * T + sensor.parametros[2] * T**2 + sensor.parametros[3] * T**3
        else:
            raise ValueError("Tipo de curva no soportado")

        plt.plot(sensor.temperaturas - 273.15, sensor.valores, "o", label="Datos")
        plt.plot(sensor.temperaturas - 273.15, y, label="Curva ajustada")
        plt.xlabel("Temperatura (°K)")
        plt.ylabel(f"{sensor.unidades_valores}")
        plt.title(f"Grafica de {sensor.nombre_sensor} con curva ajustada")
        plt.legend()
        plt.show()
    
    @staticmethod
    def grafica_simple(x, y):
        plt.plot(x, y - 273.15, "o")
    
    @staticmethod
    def grafica_basica(y):
        plt.plot(y)

class Ruido:
    def __init__(self, tipo_ruido, parametro1, parametro2=None, longitud=1000):
        self.tipo_ruido = tipo_ruido
        self.parametro1 = parametro1
        self.parametro2 = parametro2
        self.longitud = longitud
    
        self.valores = self.generarRuido()

    def generarRuido(self):
        """
        """
        if self.tipo_ruido == "gaussiano":
            ruido = np.random.normal(self.parametro1, self.parametro2, self.longitud)
        elif self.tipo_ruido == "uniforme":
            ruido = np.random.uniform(self.parametro1, self.parametro2, self.longitud)
        elif self.tipo_ruido == "exponencial":
            ruido = np.random.exponential(self.parametro1, self.longitud)
        elif self.tipo_ruido == "poisson":
            ruido = np.random.poisson(self.parametro1, self.longitud)
        
        print(self.tipo_ruido)
        return ruido

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


class Simulacion():
    def __init__(self,  horno,  lista_sensores, num_iteraciones= 5):
        self.num_iteraciones = num_iteraciones
        self.horno = horno
        self.lista_sensores = lista_sensores

    def simularSensor(self, sensor, ruido):
        """
        """
        temperaturas = self.horno.temperaturas
        valores = sensor.calcularValores(temperaturas)
        # ruido = valores * ruido
        # valores_ruido = valores + ruido
        # temperaturas_ruido = sensor.calcularTemperatura(sensor, valores_ruido)

        # return valores_ruido, temperaturas_ruido
        mean = np.mean(valores)
        std = np.std(valores)
        valores_ruido_normalizados = (valores - mean)/std + ruido 
        valores_ruido_desnormalizados = valores_ruido_normalizados * std + mean
        
        temperaturas_ruido = sensor.calcularTemperatura(sensor, valores_ruido_desnormalizados)
        return valores_ruido_desnormalizados, temperaturas_ruido
        
    
    def simulacionGaussianos(self):
        lista_sensores_horno = self.lista_sensores.copy()
        for nombreSensor, sensor in self.lista_sensores.items():
            ruido = self.generarDiccionarioRuidos()["gaussiano"]
            valores_ruido, temperata_ruido = self.simularSensor(sensor, ruido)
            lista_sensores_horno[nombreSensor] = {
                "valores_ruido": valores_ruido,
                "temperaturas_ruido": temperata_ruido
            }
        return lista_sensores_horno

    def simulacionVariosRuidos(self):
        lista_sensores_horno = self.lista_sensores.copy()
        for i, (nombreSensor, sensor) in enumerate(self.lista_sensores.items()):
            keys_ruido = list(self.generarDiccionarioRuidos().keys())
            ruido = self.generarDiccionarioRuidos()[keys_ruido[i]]

            valores_ruido, temperata_ruido = self.simularSensor(sensor, ruido)
            lista_sensores_horno[nombreSensor] = {
                "valores_ruido": valores_ruido,
                "temperaturas_ruido": temperata_ruido
            }
        return lista_sensores_horno

    def monteCarlo(self, tipo_simulacion):
        """
        Esta funcion simula los sensores utilizando Monte Carlo
        """
        simulaciones = {} 
        if tipo_simulacion == "gaussiano":
            for i in range(self.num_iteraciones):
                simulaciones[i] = self.simulacionGaussianos()
        elif tipo_simulacion == "variosRuidos":
            for i in range(self.num_iteraciones):
                simulaciones[i] = self.simulacionVariosRuidos()
        return simulaciones

    
    def generarDiccionarioRuidos(self):

        listaRuidos = {
        "gaussiano": Ruido("gaussiano", 0, 0.1, longitud).valores,
        "uniforme": Ruido("uniforme", 0, 0.1, longitud).valores,
        "exponencial": Ruido("exponencial", 0.1, None, longitud).valores,
        "poisson": Ruido("poisson", 0.1, None ,longitud).valores
        }
        return listaRuidos

def extraer_submuestra(diccionario, rango):
    """
    Esta funcion extrae una submuestra de un diccionario
    """
    
    t_min = min(diccionario.keys())
    t_max = max(diccionario.keys())
    longitud = t_max - t_min
    distancia = (longitud * rango) // 2 
    t_min = t_min + distancia
    t_max = t_max - distancia

    submuestra = {}

    for key, value in diccionario.items():
        if key >= t_min and key <= t_max:
            submuestra[key] = value
    return submuestra


sensor_PT1000 = Sensor(PT1000_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
sensor_TYPE_K = Sensor(TYPE_K_DICT, "Type K", "Voltaje (mV)", "polinomial")
sensor_TYPE_E = Sensor(TYPE_E_DICT, "Type E", "Voltaje (mV)", "polinomial")
sensor_TYPE_TMP = Sensor(TMP235Q1DICT, "TMP235-Q1", "Voltaje (mV)", "lineal")
sensor_NTCLE100E3338 = Sensor(NTCLE100E3338_DICT, "NTCLE100E3338", "Resistencia (Ohmios)", "exponencial")



# Graficas.graficar_sensor_con_curva(sensor_PT1000)
# Graficas.graficar_sensor_con_curva(sensor_TYPE_K)
# Graficas.graficar_sensor_con_curva(sensor_TYPE_E)
# Graficas.graficar_sensor_con_curva(sensor_TYPE_TMP)
# Graficas.graficar_sensor_con_curva(sensor_NTCLE100E3338)





PT1000_60_DICT = extraer_submuestra(PT1000_DICT, 0.6)
TYPE_K_60_DICT = extraer_submuestra(TYPE_K_DICT, 0.6)
TYPE_E_60_DICT = extraer_submuestra(TYPE_E_DICT, 0.6)
TYPE_TMP_60_DICT = extraer_submuestra(TMP235Q1DICT, 0.6)
NTCLE100E3338_DICT = extraer_submuestra(NTCLE100E3338_DICT, 0.6)

sensor_PT1000_60 = Sensor(PT1000_60_DICT, "PT1000", "Resistencia (Ohmios)", "lineal")
sensor_TYPE_K_60 = Sensor(TYPE_K_60_DICT, "Type K", "Voltaje (mV)", "polinomial")
sensor_TYPE_E_60 = Sensor(TYPE_E_60_DICT, "Type E", "Voltaje (mV)", "polinomial")
sensor_TYPE_TMP_60 = Sensor(TYPE_TMP_60_DICT, "TMP235-Q1", "Voltaje (mV)", "lineal")
sensor_NTCLE100E3338_60 = Sensor(NTCLE100E3338_DICT, "NTCLE100E3338", "Resistencia (Ohmios)", "exponencial")

lista_sensores = {
    "PT1000": sensor_PT1000_60,
    "TYPE_K": sensor_TYPE_K_60,
    "TYPE_E": sensor_TYPE_E_60,
    #"TYPE_TMP": sensor_TYPE_TMP_60,
    "NTCLE100E3338": sensor_NTCLE100E3338_60
}

# Puntos de la tabla con su curva
# Graficas.graficar_sensor_con_curva(sensor_PT1000_60)
# Graficas.graficar_sensor_con_curva(sensor_TYPE_K_60)
# Graficas.graficar_sensor_con_curva(sensor_TYPE_E_60)
# Graficas.graficar_sensor_con_curva(sensor_TYPE_TMP_60)
# Graficas.graficar_sensor_con_curva(sensor_NTCLE100E3338_60)


print("--------------------------")
print("Parametros de los sensores")
print("--------------------------")
print("Parametros del sensor PT1000")
print(sensor_PT1000_60.obtenerParametros())
print("Parametros sensor TYPE_K")
print(sensor_TYPE_K_60.obtenerParametros())
print("Parametros sensor TYPE_E")
print(sensor_TYPE_E_60.obtenerParametros())
print("Parametros sensor TYPE_TMP")
print(sensor_TYPE_TMP_60.obtenerParametros())
print("Parametros sensor NTCLE100E3338")
print(sensor_NTCLE100E3338_60.obtenerParametros())



print("--------------------------")
print("Errores de ajuste")
print("--------------------------")
print("PT1000:    ", Metodos.rmse(sensor_PT1000_60.valores, sensor_PT1000_60.calcularValores(sensor_PT1000_60.temperaturas)))
print("TYPE_K:    ", Metodos.rmse(sensor_TYPE_K_60.valores, sensor_TYPE_K_60.calcularValores(sensor_TYPE_K_60.temperaturas)))
print("TYPE_E:    ", Metodos.rmse(sensor_TYPE_E_60.valores, sensor_TYPE_E_60.calcularValores(sensor_TYPE_E_60.temperaturas)))
print("TYPE_TMP:  ", Metodos.rmse(sensor_TYPE_TMP_60.valores, sensor_TYPE_TMP_60.calcularValores(sensor_TYPE_TMP_60.temperaturas)))
print("NTCLE100E3338: ", Metodos.rmse(sensor_NTCLE100E3338_60.valores, sensor_NTCLE100E3338_60.calcularValores(sensor_NTCLE100E3338_60.temperaturas)))


print("--------------------------")
print("Crear Horno")
X = 100
Y = 60
Z = 30
W = 50
T0 = 0

longitud = Y + W
horno = Horno(X, Y, Z, W, T0)

#horno.graficar_perfil_temperatura()


print("--------------------------")
print("Simulacion")
print("--------------------------")

simulacion = Simulacion(horno, num_iteraciones=5, lista_sensores=lista_sensores)

print("--------------------------")
print("Simulacion con varios gaussianos")
print("--------------------------")

sensoresSimulados = simulacion.simulacionGaussianos()


# Graficas.grafica_basica(sensoresSimulados["PT1000"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimulados["TYPE_K"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimulados["TYPE_E"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimulados["TYPE_TMP"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimulados["NTCLE100E3338"]["temperaturas_ruido"] - 273.15)
# Graficas.grafica_basica(horno.temperaturas )
# plt.show()



print("--------------------------")
print("Simulacion con varios ruidos")
print("--------------------------")

sensoresSimuladosVarios = simulacion.simulacionVariosRuidos()

# Graficas.grafica_basica(sensoresSimuladosVarios["PT1000"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimuladosVarios["TYPE_K"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimuladosVarios["TYPE_E"]["temperaturas_ruido"] )
# #Graficas.grafica_basica(sensoresSimuladosVarios["TYPE_TMP"]["temperaturas_ruido"] )
# Graficas.grafica_basica(sensoresSimuladosVarios["NTCLE100E3338"]["temperaturas_ruido"] - 273.15)
# Graficas.grafica_basica(horno.temperaturas )
# plt.show()



print("--------------------------")
print("Multples simulacion con Monte Carlo")
print("--------------------------")

montecarlo = simulacion.monteCarlo("gaussiano")



print()

# Llamar a la función para graficar los histogramas
graficar_histogramas_montecarlo(montecarlo, lista_sensores)
# valores_PT1000_ruido = Ruido.ruidoGaussiano(sensor_PT1000, 1.5)
# temperaturas_PT1000_ruido = Sensor.calcularTemperatura(sensor_PT1000, valores_PT1000_ruido)

# Graficas.graficar_sensor_con_curva(sensor_PT1000)
# Graficas.grafica_simple(temperaturas_PT1000_ruido, valores_PT1000_ruido)
# print("valores del PT_1000_ con ruido", valores_PT1000_ruido)
# plt.show()
