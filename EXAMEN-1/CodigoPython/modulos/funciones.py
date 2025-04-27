
class Funciones:

    @staticmethod
    def subRango(diccionario, rango):
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