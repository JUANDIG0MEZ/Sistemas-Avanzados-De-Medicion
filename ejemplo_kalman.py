import numpy as np
import tkinter as tk


class Arena:
	def __init__(self, ancho, alto):
		self.ovalo = 5
		self.ancho = ancho
		self.alto = alto
		self.lienzo = tk.Canvas(width = ancho, height=alto)
		self.lienzo.pack()
		self.lienzo.bind("<Motion>", self.raton_mover)

	def raton_mover(self, event):
		x, y = event.x, event.y
		#self.lienzo.delete("all")
		self.lienzo.create_oval(x - self.ovalo, y-self.ovalo, x + self.ovalo, y + self.ovalo, fill="red")

		#filtrado de kalman
		# predecir
		kalman.predict()
		# actualizar
		kalman.update(np.array([[x], [y]]))

		# dibujar la salida de kalman
		est_x, est_y = float(kalman.x[0, 0]), float(kalman.x[1, 0])

		self.lienzo.create_oval(est_x - self.ovalo, est_y-self.ovalo, est_x + self.ovalo, est_y + self.ovalo, fill="blue")
		self.lienzo.update()


class Kalman:
	def __init__(self, dt):
		self.dt = dt
		# Variables que tengo que actualizar
		self.x = np.array([[0], [0], [0], [0]])
		self.P = np.eye(4)

		# Modelo dinamico
		self.F = np.eye(4) # Movimiento brawniniano

		# Modelo brawninano para Near Constant Velocity
		self.F = np.array(
			[[1, 0, dt, 0],
			 [0, 1, 0, dt],
			 [0, 0, 1, 0],
			 [0, 0, 0, 1]])

		# Ecuacion de medida
		self.H = np.array([[1, 0, 0, 0],
						   [0, 1, 0, 0]])
		# Covarianza de medida y proceso
		self.R = np.eye(2) * 5
		self.Q = np.eye(4) * 1e-5

	def predict(self):
		# Calcular el prior
		self.x = self.F @ self.x
		self.P = self.F @ self.P @ self.F.T + self.Q

	def update(self, z):
		"""
		Calcula el posterior dada la nueva medida z que acaba de llegar
		:param z: nueva medida. vector de dos elementos.
		"""
		# Calcular la innovacion
		y = z - self.H @ self.x

		# Ganancia de kalman
		s = self.H @ self.P @ self.H.T + self.R
		k = self.P @ self.H.T @ np.linalg.inv(s)

		# posterior
		self.x = self.x + k @ y

		I = np.eye(4)
		self.P = (I - k @ self.H) @ self.P




if __name__ == "__main__":
	root = tk.Tk()
	arena = Arena(800, 600)
	kalman = Kalman(0.1)
	root.mainloop()
