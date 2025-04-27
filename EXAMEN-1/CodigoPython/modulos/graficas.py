import matplotlib.pyplot as plt
import numpy as np

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

    @staticmethod
    def graficar_rangos_sensores(sensores, rango_deseado):
        """"
        Esta funcion grafica los rangos de los sensores, mediante sus keys
        y grafica el rango deseado por el usuario,
        dicha grafica solo tiene un eje de valores de grados centigrados, el otro eje es el nombre de cada sensor
        """
        j = 0
        for i, sensor in enumerate(sensores):

            nombre_sensor = sensor.nombre_sensor
            j = j+0.1
            # Graficar los valores del sensor
            plt.plot(sensor.temperaturas, np.ones(len(sensor.temperaturas))*(j), lw = 15, label=nombre_sensor)
        
        # Graficar el rango deseado
        plt.plot(np.linspace(rango_deseado[0], rango_deseado[1], 100), np.ones(100)*(j+0.1), color='r', lw = 15, label='Rango deseado')
        plt.axvline(x=rango_deseado[0], color='black', linestyle='--')
        plt.axvline(x=rango_deseado[1], color='black', linestyle='--')

        plt.xlabel("Temperatura (°C)")
        plt.ylabel("Sensores")
        #apago el eje y para que no se vea
        plt.yticks([])
        plt.title("Rangos de los sensores")
        plt.legend()
        plt.ylim(0, j+0.2)
        plt.show()
