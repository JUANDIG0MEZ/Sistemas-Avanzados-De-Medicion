import matplotlib.pyplot as plt
import numpy as np
from tablas import *

def plot_sensores(lista_sensores):
	for nombre, datos in lista_sensores.items():
		for i in datos.items():
			temperatura = i[0]
			variable = i[1]
			# Plot the data
			plt.scatter(temperatura, variable, label=f'Temperatura: {temperatura}°C')
		plt.savefig(f'sensor {nombre}.svg')
		plt.clf()

lista_sensores= {"pt_1000": pt_1000,"type_k": type_k, "type_e": type_e,"tmp235_q1_dict": tmp235_q1_dict, "ntcle100e3338_dict": ntcle100e3338_dict}
lista_rangos = {"pt_1000": (-40, 100), "type_k": (-200, 1250), "type_e": (-200, 900), "tmp235_q1_dict": (-40, 150), "ntcle100e3338_dict": (-40, 150)}
# se obtiene un rango de operacion del 60%
plot_sensores(lista_sensores)


def svd (A, b):
	U, S, VT = np.linalg.svd(A, full_matrices=False)

	S_inv = np.diag(1 / S)

	A_inv = VT.T @ S_inv @ U.T

	return A_inv @ b




def funcion_caracteristica_linea_recta(dict_sensor):
	"""
	Esta funcion sirve para pt_1000 y tmp235_q1_dict
	"""
	temperaturas = []
	valores = []
	for temperatura, valor in dict_sensor.items():
		temperaturas.append(temperatura)
		valores.append(valor)

	# Convertimos las listas a arrays de numpy
	temperaturas = np.array(temperaturas)
	valores = np.array(valores)
	# Creamos la matriz A
	A = np.array([temperaturas, np.ones(len(valores))]).T
	b = np.array(valores).T

	coeficientes = svd(A, b)
	print(coeficientes)
	return coeficientes


m, b = funcion_caracteristica_linea_recta(lista_sensores["pt_1000"])

x = np.linspace(-40, 100, 100)
y = m * x + b

plt.plot(x, y, label='Ajuste Lineal', color='red')
plt.scatter(list(lista_sensores["pt_1000"].keys()), list(lista_sensores["pt_1000"].values()), label='Datos Sensor', color='blue')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Valor Sensor')
plt.savefig('ajuste.svg')