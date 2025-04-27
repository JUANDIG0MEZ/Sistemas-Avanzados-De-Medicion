import numpy as np

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