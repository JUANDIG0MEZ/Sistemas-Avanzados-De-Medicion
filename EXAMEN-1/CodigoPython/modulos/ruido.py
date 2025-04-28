import numpy as np

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
            
        return ruido
    
    
        
