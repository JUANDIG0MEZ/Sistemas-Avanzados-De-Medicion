"""
Ejemplo de funciones anonimas en python
"""

# Una funcion anonima es aquella que no es necesario declara 
# explicitamente por nombre.
# Es decir, no se utiliza la estructura def nombre(parametros).

# Ejemplo de una funcion declarada
def f1(x, y):
    return x + y

# Ejemplo de una funcion anonima (lambda)

f2 = lambda x, y: x + y

r1 = f1(2, 3)
print(r1)

r2 = f2(2, 3)
print(r2)

