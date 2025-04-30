import numpy as np

class Ruido:
    def __init__(self, tipo_ruido, parametro1, parametro2=None, longitud=1000):
        self.tipo_ruido = tipo_ruido
        self.parametro1 = parametro1
        self.parametro2 = parametro2
        self.longitud = longitud
    
        self.valores = self.generarRuido()

    @staticmethod
    def ruido_gaussiano(media, desviacion):
        ruido = np.random.normal(media, desviacion)
        return ruido

    @staticmethod
    def ruido_uniforme(minimo, maximo):
        ruido = np.random.uniform(minimo, maximo)
        return ruido

    @staticmethod
    def ruido_poisson(lam):
        ruido = np.random.poisson(lam=lam)
        return ruido
    @staticmethod
    def ruido_cauchy(media, desviacion):
        ruido = np.random.standard_cauchy(1) * 0.05
        ruido = np.clip(ruido, -1, 1)
        return ruido
    # @staticmethod
    # def ruido_laplace(media, desviacion):
    #     ruido = np.random.laplace(parametro1, parametro2, 1)
    #     return ruido
    
    @staticmethod
    def generarOutlier(probabilidad):
        if np.random.rand() < probabilidad:
            if np.random.rand() < 0.5:
                outlier_value = np.random.uniform(1.5, 3)
            else:
                outlier_value = np.random.uniform(-1.5, -3)
        else: 
            outlier_value = 0
        return outlier_value
    
        
