import matplotlib.pyplot as plt
import numpy as np
import os
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
        y = sensor.calcularValores(sensor.temperaturas)

        plt.plot(sensor.temperaturas, sensor.valores, "o", label="Datos")
        plt.plot(sensor.temperaturas, y, label="Curva ajustada")
        plt.xlabel("Temperatura (°C)")
        plt.ylabel(f"{sensor.unidades_valores}")
        plt.title(f"Gráfica de {sensor.nombre_sensor}")
        plt.legend()
        plt.savefig(f"imagenes/{sensor.nombre_sensor}_con_ajuste.png")
        plt.show()
    
    @staticmethod
    def grafica_xy(x, y, show=True, estilo='o', title="", xlabel="", ylabel=""):
        plt.plot(x, y, estilo)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if show:
            plt.show()
    
    @staticmethod
    def grafica_y(y, show=True, estilo='o', title="", ylabel="", xlabel="", color='blue', save=False, nombre=None):
        plt.plot(y, estilo, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        if save:
            plt.savefig(f"imagenes/{nombre}")


        if show:
            plt.show()

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
        plt.savefig(f"imagenes/rangos_sensores.png")
        plt.show()

    @staticmethod
    def graficar_xy_con_error(x, y, error, show=True):
        plt.plot(x, y, color='gray')
        plt.fill_between(x, y - error, y + error, color='red')

        plt.title('Ajuste Continuo con Área de Error')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        if show:
            plt.show()
    
    @staticmethod
    def graficar_y_con_error(y, error, show=True):

        print(y)
        plt.plot(y, color='gray')
        plt.fill_between(range(len(y)), y - error, y + error, color='red')

        plt.title('Ajuste Continuo con Área de Error')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid()
        if show:
            plt.show()
