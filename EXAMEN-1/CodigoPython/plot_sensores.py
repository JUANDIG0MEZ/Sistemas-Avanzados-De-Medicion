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
# plot_sensores(lista_sensores)


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

# plt.plot(x, y, label='Ajuste Lineal', color='red')
# plt.scatter(list(lista_sensores["pt_1000"].keys()), list(lista_sensores["pt_1000"].values()), label='Datos Sensor', color='blue')
# plt.xlabel('Temperatura (°C)')
# plt.ylabel('Valor Sensor')
# plt.title('Ajuste Lineal para Sensor pt_1000')
# plt.show()
# plt.savefig('ajuste.svg')




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
        return a * np.log10(b * x) + c
        
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
    # plt.figure(figsize=(10, 6))
    # plt.scatter(temperaturas, valores, label='Datos originales', color='blue', marker='o')
    
    # Obtener rango de operación del sensor
    rango_min, rango_max = rango_sensor
    rango_operativo = rango_max - rango_min
    
    # Calcular rango efectivo (60% del rango operativo, centrado)
    centro_rango = (rango_max + rango_min) / 2
    rango_efectivo = rango_operativo * 1
    rango_efectivo_min = centro_rango - rango_efectivo/2
    rango_efectivo_max = centro_rango + rango_efectivo/2
    
    # Dibujar líneas para marcar rango efectivo
    # plt.axvline(x=rango_efectivo_min, color='r', linestyle='--', alpha=0.5, label='Rango efectivo (60%)')
    # plt.axvline(x=rango_efectivo_max, color='r', linestyle='--', alpha=0.5)
    
    

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
        # plt.plot(temp_curve, valores_curve, 'r-', label=etiqueta_ajuste)
        
        # # Agregar ecuación como texto en la gráfica
        # plt.text(0.05, 0.95, etiqueta_ajuste, transform=plt.gca().transAxes, 
        #          verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
    except Exception as e:
        plt.text(0.05, 0.95, f"Error en ajuste: {str(e)}", transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # # Configurar gráfica
    # plt.title(f'Sensor {nombre_sensor} con ajuste {tipo_ajuste}')
    # plt.xlabel('Temperatura (°C)')
    # plt.ylabel('Valor del Sensor')
    # plt.grid(True, alpha=0.3)
    # plt.legend()
    
    # Mostrar y guardar
    # plt.tight_layout()
    # if guardar:
    #     plt.savefig(f'sensor_{nombre_sensor}_{tipo_ajuste}.svg')
    # # plt.show()
    # plt.clf()

    # Almacenar información adicional para retornar
    info_ajuste = {
        'params': params,
        'tipo_ajuste': tipo_ajuste,
        'funcion': funcion
    }
    
    # Devolver los parámetros del ajuste
    return info_ajuste

# Ejemplo de uso
# Para ajuste lineal (por defecto)
# plot_sensores_ajuste(lista_sensores["pt_1000"], lista_rangos["pt_1000"], nombre_sensor='pt_1000')
# plot_sensores_ajuste(lista_sensores["tmp235_q1_dict"], lista_rangos["tmp235_q1_dict"], nombre_sensor='tmp235_q1_dict')
# Para ajuste exponencial
# plot_sensores_ajuste(lista_sensores["type_k"], lista_rangos["type_k"], tipo_ajuste='exponential', nombre_sensor='type_k')
# plot_sensores_ajuste(lista_sensores["type_e"], lista_rangos["type_e"], tipo_ajuste='exponential', nombre_sensor='type_e')
# # Para ajuste logarítmico
# plot_sensores_ajuste(lista_sensores["ntcle100e3338_dict"], lista_rangos["ntcle100e3338_dict"], tipo_ajuste='logarithmic', nombre_sensor='ntcle100e3338_dict')

# # Para ajuste polinomial
# plot_sensores_ajuste(lista_sensores, lista_rangos, tipo_ajuste='polynomial')


def adquisicion(sensor, temperatura):
    """
    Esta funcion sirve para pt_1000 y tmp235_q1_dict
    """
    # Convertimos las listas a arrays de numpy
    temperaturas = np.array(list(sensor.keys()))
    valores = np.array(list(sensor.values()))
    
    # Creamos la matriz A
    A = np.array([temperaturas, np.ones(len(valores))]).T
    b = np.array(valores).T
    
    coeficientes = svd(A, b)
    
    # Calculamos el valor del sensor para la temperatura dada
    valor_sensor = coeficientes[0] * temperatura + coeficientes[1]
    temperatura_leida = (valor_sensor - coeficientes[1]) / coeficientes[0]

    # Devolvemos el valor del sensor y la temperatura leída
    print(f"Valor sensor: {valor_sensor}, Temperatura leída: {temperatura_leida}")

    
    return valor_sensor, temperatura_leida

# adquisicion(lista_sensores["pt_1000"], 25)


def simular_horno(sensores_seleccionados, info_ajustes, temperatura_inicial=25, temperatura_final=250, 
                  tiempo_total=3600, muestras=300, nivel_ruido=0.05, mostrar_grafica=True):
    """
    Simula la medición de temperatura dentro de un horno usando diferentes sensores.
    
    Parámetros:
    - sensores_seleccionados: Lista de nombres de sensores a simular
    - info_ajustes: Diccionario con información de ajuste para cada sensor
    - temperatura_inicial: Temperatura inicial del horno en °C
    - temperatura_final: Temperatura final del horno en °C
    - tiempo_total: Tiempo total de simulación en segundos
    - muestras: Cantidad de muestras a tomar durante la simulación
    - nivel_ruido: Nivel de ruido no gaussiano (como porcentaje del rango)
    - mostrar_grafica: Si se muestra la gráfica de la simulación
    
    Retorna:
    - Diccionario con los datos simulados para cada sensor
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import skewnorm  # Para generar ruido no gaussiano (sesgado)
    import time
    
    # Crear vector de tiempo
    tiempos = np.linspace(0, tiempo_total, muestras)
    
    # Simular perfil de temperatura del horno (rampa + estabilización)
    tiempo_calentamiento = tiempo_total * 0.4  # 40% del tiempo para calentar
    
    temperaturas_reales = []
    for t in tiempos:
        if t < tiempo_calentamiento:
            # Rampa de calentamiento
            temp = temperatura_inicial + (temperatura_final - temperatura_inicial) * (t / tiempo_calentamiento)
        else:
            # Estabilización con pequeñas fluctuaciones (simulando control PID)
            fluctuacion = np.sin(t/100) * 3  # Fluctuación de ±3°C
            temp = temperatura_final + fluctuacion
        
        temperaturas_reales.append(temp)
    
    temperaturas_reales = np.array(temperaturas_reales)
    
    # Función para aplicar los diferentes modelos de sensores
    def aplicar_modelo_sensor(temperatura, info_ajuste):
        tipo = info_ajuste['tipo_ajuste']
        params = info_ajuste['params']
        funcion = info_ajuste['funcion']
        
        # Calcular la salida del sensor según el modelo
        valor_sensor = funcion(temperatura, *params)
        

            
        return valor_sensor
    
    # Función para generar ruido no gaussiano
    def generar_ruido_no_gaussiano(amplitud, tamaño, sesgo=5):
        """Genera ruido con distribución sesgada (no gaussiana)"""
        # Parámetro de sesgo (positivo = sesgo a la derecha, negativo = sesgo a la izquierda)
        ruido_base = skewnorm.rvs(a=sesgo, loc=0, scale=amplitud, size=tamaño)
        return ruido_base
    
    # Simular lecturas de cada sensor
    resultados_simulacion = {}
    
    for nombre_sensor in sensores_seleccionados:
        if nombre_sensor not in info_ajustes:
            print(f"No se encontró información de ajuste para el sensor {nombre_sensor}")
            continue
            
        # Obtener las lecturas ideales según el modelo del sensor
        valores_ideales = aplicar_modelo_sensor(temperaturas_reales, info_ajustes[nombre_sensor])
        
        # Calcular amplitud del ruido basado en el rango de valores
        rango_valores = np.max(valores_ideales) - np.min(valores_ideales)
        amplitud_ruido = rango_valores * nivel_ruido
        
        # Generar ruido no gaussiano
        ruido = generar_ruido_no_gaussiano(amplitud_ruido, len(temperaturas_reales))
        
        # Añadir ruido a las lecturas del sensor
        valores_con_ruido = valores_ideales + ruido
        
        # Almacenar resultados
        resultados_simulacion[nombre_sensor] = {
            'tiempos': tiempos,
            'temperaturas_reales': temperaturas_reales,
            'valores_ideales': valores_ideales,
            'valores_medidos': valores_con_ruido,
            'ruido': ruido
        }
    
    # Visualizar resultados
    if mostrar_grafica:
        plt.figure(figsize=(15, 10))
        
        # Gráfica 1: Temperatura real del horno vs tiempo
        plt.subplot(2, 1, 1)
        plt.plot(tiempos / 60, temperaturas_reales, 'k-', linewidth=2, label='Temperatura real')
        plt.title('Simulación de temperatura en horno')
        plt.xlabel('Tiempo (minutos)')
        plt.ylabel('Temperatura (°C)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Gráfica 2: Valores medidos por cada sensor
        plt.subplot(2, 1, 2)
        colores = ['b', 'r', 'g', 'm', 'c', 'y']
        
        for i, nombre_sensor in enumerate(sensores_seleccionados):
            if nombre_sensor in resultados_simulacion:
                data = resultados_simulacion[nombre_sensor]
                color = colores[i % len(colores)]
                plt.plot(data['tiempos'] / 60, data['valores_medidos'], color, 
                         label=f'Sensor {nombre_sensor}', alpha=0.7)
        
        plt.title('Valores medidos por los sensores (con ruido)')
        plt.xlabel('Tiempo (minutos)')
        plt.ylabel('Valor del sensor')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('simulacion_horno.svg')
        plt.show()
        
        # Gráfica adicional: Distribución del ruido para cada sensor
        plt.figure(figsize=(12, 8))
        
        for i, nombre_sensor in enumerate(sensores_seleccionados):
            if nombre_sensor in resultados_simulacion:
                plt.subplot(2, 2, i+1)
                data = resultados_simulacion[nombre_sensor]
                plt.hist(data['ruido'], bins=30, alpha=0.7, color=colores[i % len(colores)])
                plt.title(f'Distribución del ruido - {nombre_sensor}')
                plt.xlabel('Magnitud del ruido')
                plt.ylabel('Frecuencia')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('distribucion_ruido.svg')
        plt.show()
    
    return resultados_simulacion

# Función adicional para convertir los valores medidos de vuelta a temperatura
def convertir_a_temperatura(valores_medidos, info_ajuste):
    """
    Convierte los valores medidos por el sensor a temperaturas estimadas
    utilizando el modelo inverso.
    
    Parámetros:
    - valores_medidos: Array con los valores medidos por el sensor
    - info_ajuste: Información del ajuste del sensor
    
    Retorna:
    - Array con las temperaturas estimadas
    """
    import numpy as np
    from scipy.optimize import fsolve
    
    tipo = info_ajuste['tipo_ajuste']
    params = info_ajuste['params']
    funcion = info_ajuste['funcion']
    
    valores_normalizados = valores_medidos
    
    # Función para resolver el modelo inverso (encontrar T para un valor dado)
    def resolver_inversa(valor_medido):
        def ecuacion(temp):
            return funcion(temp, *params) - valor_medido
        
        # Solución inicial aproximada
        temp_inicial = 25  # Temperatura ambiente como punto de partida
        
        # Resolver la ecuación
        try:
            temp_estimada = fsolve(ecuacion, temp_inicial)[0]
            return temp_estimada
        except:
            return np.nan
    
    # Convertir cada valor medido a temperatura
    temperaturas_estimadas = []
    for valor in valores_normalizados:
        temp = resolver_inversa(valor)
        temperaturas_estimadas.append(temp)
    
    return np.array(temperaturas_estimadas)

# Función para analizar y visualizar la precisión de los sensores
def analizar_precision(resultados_simulacion, info_ajustes):
    """
    Analiza la precisión de los sensores, comparando las temperaturas estimadas
    con las temperaturas reales.
    
    Parámetros:
    - resultados_simulacion: Resultados de la simulación
    - info_ajustes: Información de ajuste de cada sensor
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Calcular temperaturas estimadas para cada sensor
    for nombre_sensor, datos in resultados_simulacion.items():
        if nombre_sensor in info_ajustes:
            # Convertir valores medidos a temperaturas
            temperaturas_estimadas = convertir_a_temperatura(
                datos['valores_medidos'], info_ajustes[nombre_sensor])
            
            # Calcular error
            error = temperaturas_estimadas - datos['temperaturas_reales']
            
            # Guardar en resultados
            resultados_simulacion[nombre_sensor]['temperaturas_estimadas'] = temperaturas_estimadas
            resultados_simulacion[nombre_sensor]['error'] = error
    
    # Visualizar resultados
    plt.figure(figsize=(15, 12))
    
    # Gráfica 1: Temperatura real vs estimada
    plt.subplot(2, 1, 1)
    colores = ['b', 'r', 'g', 'm']
    
    # Temperatura real (referencia)
    tiempos = resultados_simulacion[list(resultados_simulacion.keys())[0]]['tiempos']
    temp_real = resultados_simulacion[list(resultados_simulacion.keys())[0]]['temperaturas_reales']
    plt.plot(tiempos / 60, temp_real, 'k-', linewidth=2, label='Temperatura real')
    
    # Temperaturas estimadas por cada sensor
    for i, (nombre_sensor, datos) in enumerate(resultados_simulacion.items()):
        if 'temperaturas_estimadas' in datos:
            color = colores[i % len(colores)]
            plt.plot(tiempos / 60, datos['temperaturas_estimadas'], color, 
                     linestyle='--', alpha=0.7, label=f'Estimada {nombre_sensor}')
    
    plt.title('Temperatura real vs Temperatura estimada')
    plt.xlabel('Tiempo (minutos)')
    plt.ylabel('Temperatura (°C)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfica 2: Error de cada sensor
    plt.subplot(2, 1, 2)
    
    for i, (nombre_sensor, datos) in enumerate(resultados_simulacion.items()):
        if 'error' in datos:
            color = colores[i % len(colores)]
            plt.plot(tiempos / 60, datos['error'], color, 
                     label=f'Error {nombre_sensor}')
    
    plt.title('Error de medición (Temperatura estimada - Temperatura real)')
    plt.xlabel('Tiempo (minutos)')
    plt.ylabel('Error (°C)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.legend()
    
    
    labels = []
    rmse_values = []
    max_error_values = []
    
    for nombre_sensor, datos in resultados_simulacion.items():
        if 'error' in datos:
            # Calcular RMSE
            rmse = np.sqrt(np.mean(np.square(datos['error'])))
            # Calcular error máximo
            max_error = np.max(np.abs(datos['error']))
            
            labels.append(nombre_sensor)
            rmse_values.append(rmse)
            max_error_values.append(max_error)
            
            print(f"Sensor {nombre_sensor}:")
            print(f"  RMSE: {rmse:.4f} °C")
            print(f"  Error máximo: {max_error:.4f} °C")
            print(f"  Error promedio: {np.mean(np.abs(datos['error'])):.4f} °C")
            print()
    
    # Gráfico de barras con métricas de error
    x = np.arange(len(labels))
    width = 0.35

    fig3, ax = plt.subplots(figsize=(10, 6))
    bar1 = ax.bar(x - width/2, rmse_values, width, label='RMSE (°C)')
    bar2 = ax.bar(x + width/2, max_error_values, width, label='Error Máximo (°C)')
    
    ax.set_ylabel('Error (°C)')
    ax.set_title('Métricas de Error por Sensor')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('metricas_error.svg')
    plt.show()
    
    return


# 
# 
# Para ajuste exponencial
# 
# plot_sensores_ajuste(lista_sensores["type_e"], lista_rangos["type_e"], tipo_ajuste='exponential', nombre_sensor='type_e')
# # Para ajuste logarítmico
# 



# Inicializa un diccionario para guardar todos los parámetros
info_ajustes = {}

# Guarda los resultados de cada llamada a plot_sensores_ajuste
info_ajustes['pt_1000'] = plot_sensores_ajuste(
    lista_sensores["pt_1000"], 
    lista_rangos["pt_1000"], 
    nombre_sensor="pt_1000"
)

info_ajustes['tmp235_q1'] = plot_sensores_ajuste(
    lista_sensores["tmp235_q1_dict"], 
    lista_rangos["tmp235_q1_dict"], 
    nombre_sensor="tmp235_q1"
)

info_ajustes['type_k'] = plot_sensores_ajuste(
    lista_sensores["type_k"], 
    lista_rangos["type_k"], 
    tipo_ajuste='exponential', 
    nombre_sensor="type_k"
)

info_ajustes['ntcle100e3338'] = plot_sensores_ajuste(
    lista_sensores["ntcle100e3338_dict"], 
    lista_rangos["ntcle100e3338_dict"], 
    tipo_ajuste='logarithmic', 
    nombre_sensor="ntcle100e3338", 

)

# 2. Seleccionar los 4 sensores más fáciles de modelar
sensores_seleccionados = ['pt_1000', 'tmp235_q1', 'type_k', 'ntcle100e3338']

# 3. Realizar la simulación del horno
resultados = simular_horno(
    sensores_seleccionados=sensores_seleccionados,
    info_ajustes=info_ajustes,
    temperatura_inicial=25,       # Temperatura ambiente
    temperatura_final=250,        # Temperatura máxima del horno
    tiempo_total=3600,            # 1 hora de simulación
    muestras=300,                 # 300 mediciones
    nivel_ruido=0.03              # 3% de ruido no gaussiano
)

# 4. Analizar la precisión de los sensores
analizar_precision(resultados, info_ajustes)