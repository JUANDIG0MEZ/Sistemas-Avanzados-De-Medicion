import numpy as np

class Errores:

    @staticmethod
    def calcularErrorMedida(sensor, temperatura, valor, rmse=None):
        if sensor.nombre_sensor in ["PT1000", "TYPE_E", "TMP235"]:
            errorTemperatura = Errores.error_sobre_temperatura(sensor, temperatura)
            error = Errores.propagacionError(sensor.tipo_curva, sensor.parametros, errorTemperatura, temperatura )
        elif sensor.nombre_sensor in ["NTCLE100E3"]:
            error = Errores.error_sobre_valor(sensor, valor)
        else:

            raise ValueError("Sensor no soportado")
        
        if rmse is not None:
            error = Errores.sumaErrores(error, rmse)
        return error
        

    @staticmethod
    def error_sobre_temperatura( sensor, temperatura):
        nombre_sensor = sensor.nombre_sensor
        if nombre_sensor == "PT1000":
            homocedastico = 0.5
            return homocedastico
        
        elif nombre_sensor == "TMP235":
            homocedastico = 2.5
            return homocedastico
            
        elif nombre_sensor == "TYPE_E":
            # error sobre 0
            homocedastico_sobre = 2.2
            heterocedastico_sobre = 0.5/ 100

            # error debajo de 0
            homocedastico_debajo = 2.2
            heterocedastico_debajo = 2.0 / 100


            if temperatura < 0:
                if abs(temperatura * heterocedastico_debajo) > homocedastico_debajo:
                    return abs(temperatura * heterocedastico_debajo)
                else:
                    return homocedastico_debajo
            else:
                if abs(temperatura * heterocedastico_sobre) > homocedastico_sobre:
                    return abs(temperatura * heterocedastico_sobre)
                else:
                    return homocedastico_sobre
        else:
            raise ValueError("Sensor no soportado")
        


    @staticmethod
    def error_sobre_valor(sensor, valor):
        nombre_sensor = sensor.nombre_sensor
        if nombre_sensor == "NTCLE100E3":
            heterocedastico = 2.0 / 100
            return heterocedastico * valor
        else:
            raise ValueError("Error sobre valor: Sensor no soportado")

    @staticmethod
    def sumaErrores(error_ajuste, error_medida):
        """
        Esta funcion suma los errores de ajuste y de medida
        """
        return np.sqrt(error_ajuste**2 + error_medida**2)


    @staticmethod
    def propagacionError(tipo, parametros, errorFabricanteTemperatura, temperatura):
        """
        Esta funcion calcula el error de la resistencia a partir de la temperatura
        """
        if tipo == "lineal":
            """
            V(T) = m*T + b
            """
            m = parametros[0]
            b = parametros[1]
            error = errorFabricanteTemperatura * m
            return abs(error)

        elif tipo == "exponencial":
            """
            R(T) = A * e^(B/T)
            """
            A = np.exp(parametros[1])
            B = parametros[0]
            error = abs(errorFabricanteTemperatura * A * np.exp(B / (temperatura + 273.15)))
            return error

        elif tipo == "polinomial":
            """
            V(T0) = A + B * T + C * T^2 + D * T^3
            """
            A = parametros[0]
            B = parametros[1]
            C = parametros[2]
            D = parametros[3]
            error = abs(errorFabricanteTemperatura * (B + 2 * C * temperatura + 3 * D * temperatura**2))
            return error
        else: 
            raise ValueError("Tipo de curva no soportada")
    
