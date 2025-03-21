import numpy as np

def solucionPunto1():
     # Punto 1
     A = [[1, 2, 3, 4],
          [2, 3, 5, 7],
          [4, 1, 2, 6],
          [3, 4, 1, 5],
          [5, 6, 4, 3],
          [7, 8, 6, 2]]

     # Descomposición en valores singulares (SVD)
     U, S, Vt = np.linalg.svd(A)


     # Resolver Ax = 0 usando los valores singulares
     # La solución no trivial está en el último vector de V (correspondiente al menor valor singular)
     x_svd = Vt[-1]

     print(x_svd)


solucionPunto1()