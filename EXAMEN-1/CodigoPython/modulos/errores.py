import numpy as np

class Errores:

    @staticmethod
    def error_sobre_temperatura( nombre_sensor, temperatura):
        if nombre_sensor == "PT1000":
            homocedastico = 0.5
            return homocedastico
        elif nombre_sensor == "TYPE_K":
            # error sobre 0
            homocedastico_sobre = 2.2
            heterocedastico_sobre = 0.5/ 100

            # error debajo de 0
            homocedastico_debajo = 2.2
            heterocedastico_debajo = 2.0 / 100


            if temperatura < 0:
                if temperatura * heterocedastico_debajo > homocedastico_debajo:
                    return temperatura * heterocedastico_debajo
                else:
                    return homocedastico_debajo
            else:
                if temperatura * heterocedastico_sobre > homocedastico_sobre:
                    return temperatura * heterocedastico_sobre
                else:
                    return homocedastico_sobre
            
        elif nombre_sensor == "TYPE_E":
            # error sobre 0
            homocedastico_sobre = 2.2
            heterocedastico_sobre = 0.5/ 100

            # error debajo de 0
            homocedastico_debajo = 2.2
            heterocedastico_debajo = 2.0 / 100


            if temperatura < 0:
                if temperatura * heterocedastico_debajo > homocedastico_debajo:
                    return temperatura * heterocedastico_debajo
                else:
                    return homocedastico_debajo
            else:
                if temperatura * heterocedastico_sobre > homocedastico_sobre:
                    return temperatura * heterocedastico_sobre
                else:
                    return homocedastico_sobre
        else:
            raise ValueError("Sensor no soportado")
        


    @staticmethod
    def error_sobre_valor(nombre_sensor, valor):
        if nombre_sensor == "NTCLE100E3":
            heterocedastico = 2.0 / 100
            return heterocedastico * valor
        else:
            raise ValueError("Sensor no soportado")

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
            return error

        elif tipo == "exponencial":
            """
            R(T) = A * e^(B/T)
            """
            A = np.exp(parametros[1])
            B = parametros[0]
            error = errorFabricanteTemperatura * A * np.exp(B / (temperatura + 273.15))
            return error

        elif tipo == "polinomial":
            """
            V(T0) = A + B * T + C * T^2 + D * T^3
            """
            A = parametros[0]
            B = parametros[1]
            C = parametros[2]
            D = parametros[3]
            error = errorFabricanteTemperatura * (B + 2 * C * temperatura + 3 * D * temperatura**2)
            return error
        else: 
            raise ValueError("Tipo de curva no soportada")
    
