import matplotlib.pyplot as plt


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