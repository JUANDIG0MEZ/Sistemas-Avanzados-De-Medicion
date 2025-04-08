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
		
		plt.title(f'Gráfica de {nombre}')
		plt.xlabel('Temperatura (°C)')
		plt.ylabel('Valor Sensor')
		plt.show()
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
plt.title('Ajuste Lineal para Sensor pt_1000')
plt.show()
plt.savefig('ajuste.svg')




def plot_sensores_ajuste(datos_sensor, rango_sensor, tipo_ajuste='linear', guardar=True, nombre_sensor=None):
    """
    Genera gráficas para un sensor con ajustes de curva.
    
    Parámetros:
    - datos_sensor: Diccionario con los datos del sensor {temperatura: valor}
    - rango_sensor: Tupla con el rango de operación del sensor (min, max)
    - tipo_ajuste: Tipo de ajuste ('linear', 'exponential', 'logarithmic', 'polynomial')
    - guardar: Booleano para indicar si se guarda la gráfica
    - nombre_sensor: Nombre del sensor para la gráfica (opcional)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    
    # Definir funciones de ajuste
    def func_linear(x, a, b):
        return a*x + b
        
    def func_exponential(x, a, b, c):
        return a * np.exp(b * x) + c
        
    def func_logarithmic(x, a, b, c):
        return a * np.log(b * x) + c
        
    def func_polynomial(x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d
    
    # Mapear nombres de función a las funciones
    funciones_ajuste = {
        'linear': func_linear,
        'exponential': func_exponential,
        'logarithmic': func_logarithmic,
        'polynomial': func_polynomial
    }
    
    # Verificar que el tipo de ajuste sea válido
    if tipo_ajuste not in funciones_ajuste:
        raise ValueError(f"Tipo de ajuste '{tipo_ajuste}' no válido. Opciones: {list(funciones_ajuste.keys())}")
    
    # Función seleccionada
    funcion = funciones_ajuste[tipo_ajuste]
    
    # Detectar el nombre del sensor si no se proporcionó
    if nombre_sensor is None:
        nombre_sensor = "sensor"
    
    # Preparar datos para el ajuste
    temperaturas = []
    valores = []
    
    for temp, valor in datos_sensor.items():
        temperaturas.append(temp)
        valores.append(valor)
    
    temperaturas = np.array(temperaturas)
    valores = np.array(valores)
    
    # Ordenar datos para la gráfica
    indices_orden = np.argsort(temperaturas)
    temperaturas_ordenadas = temperaturas[indices_orden]
    valores_ordenados = valores[indices_orden]
    
    # Generar gráfica de dispersión
    plt.figure(figsize=(10, 6))
    plt.scatter(temperaturas, valores, label='Datos originales', color='blue', marker='o')
    
    # Obtener rango de operación del sensor
    rango_min, rango_max = rango_sensor
    rango_operativo = rango_max - rango_min
    
    # Calcular rango efectivo (60% del rango operativo, centrado)
    centro_rango = (rango_max + rango_min) / 2
    rango_efectivo = rango_operativo * 0.6
    rango_efectivo_min = centro_rango - rango_efectivo/2
    rango_efectivo_max = centro_rango + rango_efectivo/2
    
    # Dibujar líneas para marcar rango efectivo
    plt.axvline(x=rango_efectivo_min, color='r', linestyle='--', alpha=0.5, label='Rango efectivo (60%)')
    plt.axvline(x=rango_efectivo_max, color='r', linestyle='--', alpha=0.5)
    
    # Realizar ajuste de curva
    try:
        if tipo_ajuste == 'linear':
            params, _ = curve_fit(funcion, temperaturas, valores)
            a, b = params
            etiqueta_ajuste = f'Ajuste {tipo_ajuste}: y = {a:.4f}x + {b:.4f}'
        elif tipo_ajuste == 'exponential':
            # Valores iniciales para ayudar a la convergencia
            p0 = [1.0, 0.01, 0]
            params, _ = curve_fit(funcion, temperaturas, valores, p0=p0, maxfev=10000)
            a, b, c = params
            etiqueta_ajuste = f'Ajuste {tipo_ajuste}: y = {a:.4f}*exp({b:.4f}*x) + {c:.4f}'
        elif tipo_ajuste == 'logarithmic':
            # Asegurar que todos los valores sean positivos para logaritmo
            temp_ajuste = np.maximum(temperaturas, 0.1)
            p0 = [1.0, 1.0, 0]
            params, _ = curve_fit(funcion, temp_ajuste, valores, p0=p0, maxfev=10000)
            a, b, c = params
            etiqueta_ajuste = f'Ajuste {tipo_ajuste}: y = {a:.4f}*ln({b:.4f}*x) + {c:.4f}'
            # Usar temperaturas positivas para el ajuste mostrado
            temperaturas_ordenadas = np.maximum(temperaturas_ordenadas, 0.1)
        elif tipo_ajuste == 'polynomial':
            params, _ = curve_fit(funcion, temperaturas, valores)
            a, b, c, d = params
            etiqueta_ajuste = f'Ajuste {tipo_ajuste}: y = {a:.4f}x³ + {b:.4f}x² + {c:.4f}x + {d:.4f}'
        
        # Generar curva ajustada para visualización
        temp_curve = np.linspace(min(temperaturas), max(temperaturas), 500)
        valores_curve = funcion(temp_curve, *params)
        
        # Dibujar curva ajustada
        plt.plot(temp_curve, valores_curve, 'r-', label=etiqueta_ajuste)
        
        # Agregar ecuación como texto en la gráfica
        plt.text(0.05, 0.95, etiqueta_ajuste, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
    except Exception as e:
        plt.text(0.05, 0.95, f"Error en ajuste: {str(e)}", transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Configurar gráfica
    plt.title(f'Sensor {nombre_sensor} con ajuste {tipo_ajuste}')
    plt.xlabel('Temperatura (°C)')
    plt.ylabel('Valor del Sensor')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Mostrar y guardar
    plt.tight_layout()
    if guardar:
        plt.savefig(f'sensor_{nombre_sensor}_{tipo_ajuste}.svg')
    plt.show()
    plt.clf()
    
    # Devolver los parámetros del ajuste
    return params

# Ejemplo de uso
# Para ajuste lineal (por defecto)
plot_sensores_ajuste(lista_sensores["pt_1000"], lista_rangos["pt_1000"])
plot_sensores_ajuste(lista_sensores["tmp235_q1_dict"], lista_rangos["tmp235_q1_dict"])
# Para ajuste exponencial
plot_sensores_ajuste(lista_sensores["type_k"], lista_rangos["type_k"], tipo_ajuste='exponential')
plot_sensores_ajuste(lista_sensores["type_e"], lista_rangos["type_e"], tipo_ajuste='exponential')
# # Para ajuste logarítmico
plot_sensores_ajuste(lista_sensores["ntcle100e3338_dict"], lista_rangos["ntcle100e3338_dict"], tipo_ajuste='logarithmic')

# # Para ajuste polinomial
# plot_sensores_ajuste(lista_sensores, lista_rangos, tipo_ajuste='polynomial')