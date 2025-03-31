import numpy as np


# Definir el modelo
def modelo(x, a, b, c, d):
	return a * np.sin(b * x) + c * np.cos(d * x)


# Parámetros reales
a_real, b_real, c_real, d_real = 2.5, 1.5, 1.0, 0.8
rango = (0, 10)

# Generar datos
x = np.linspace(rango[0], rango[1], 200)
y = modelo(x, a_real, b_real, c_real, d_real)

# Inicializar parámetros
np.random.seed(42)
a, b, c, d = np.random.randn(4)

# Hiperparámetros
learning_rate = 0.01
epochs = 1000

# Gradiente descendiente
for epoch in range(epochs):
	y_pred = modelo(x, a, b, c, d)
	error = y_pred - y

	# Calcular gradientes
	grad_a = np.sum(2 * error * np.sin(b * x)) / len(x)
	grad_b = np.sum(2 * error * a * x * np.cos(b * x)) / len(x)
	grad_c = np.sum(2 * error * np.cos(d * x)) / len(x)
	grad_d = np.sum(-2 * error * c * x * np.sin(d * x)) / len(x)

	# Actualizar parámetros
	a -= learning_rate * grad_a
	b -= learning_rate * grad_b
	c -= learning_rate * grad_c
	d -= learning_rate * grad_d

	# Imprimir pérdida cada 100 iteraciones
	if epoch % 100 == 0:
		loss = np.mean(error ** 2)
		print(f"Epoch {epoch}, Loss: {loss:.4f}, a: {a:.4f}, b: {b:.4f}, c: {c:.4f}, d: {d:.4f}")

# Resultados finales
print(f"Parámetros finales: a={a:.4f}, b={b:.4f}, c={c:.4f}, d={d:.4f}")
