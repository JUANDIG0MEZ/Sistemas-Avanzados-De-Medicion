import numpy as np
import tkinter as tk
import time



class Arena:
	def __init__(self, root, width, height):
		self.kalman_NCV = None
		self.kalman_BM = None

		self.history = []
		self.predictions_NCV = []
		self.predictions_BM = []

		self.ovalo = 4
		self.ancho = width
		self.alto = height
		self.lienzo = tk.Canvas(root, width = width, height=height)
		self.lienzo.pack()
		self.lienzo.bind("<Motion>", self.move_mouse)
		self.last_time = time.time()

		self.samples = 0


	def move_mouse(self, event):
		self.lienzo.delete("all")

		# Time between frames/events
		current_time = time.time()
		dt = current_time - self.last_time
		self.last_time = current_time

		x, y = float(event.x), float(event.y)
		self.lienzo.create_oval(x - self.ovalo, y-self.ovalo, x + self.ovalo, y + self.ovalo, fill="red")
		
		if self.samples == 0:
			# Init the kalman filter
			x_init = np.array([x, y, 0, 0])  # x, y, vx, vy
			self.kalman_NCV = Kalman("NCV", x_init)
			self.kalman_BM  = Kalman("BM", x_init)
		else: 
			self.kalman_NCV.predict(dt)
			self.kalman_BM.predict(dt)

			z = np.array([[x], [y]])
			self.history.append((x, y))

			self.kalman_NCV.update(z)
			self.kalman_BM.update(z)
			
			x_backup_NCV, x_backup_BM = self.kalman_NCV.x.copy() , self.kalman_BM.x.copy()
			p_backup_NCV, p_backup_BM = self.kalman_NCV.P.copy() , self.kalman_BM.P.copy()
			
			for i in range(5):
				self.kalman_NCV.predict(dt)
				self.kalman_BM.predict(dt)

				x_NCV, y_NCV = float(self.kalman_NCV.x[0, 0]), float(self.kalman_NCV.x[1, 0])
				x_BM , y_BM  = float(self.kalman_BM.x[0, 0]), float(self.kalman_BM.x[1, 0])

				if i == 0:
					self.predictions_NCV.append((x_NCV, y_NCV))
					self.predictions_BM.append((x_BM, y_BM))


				self.lienzo.create_oval(x_NCV - self.ovalo, y_NCV-self.ovalo, x_NCV + self.ovalo, y_NCV + self.ovalo, fill="blue")
				self.lienzo.create_oval(x_BM - self.ovalo, y_BM-self.ovalo, x_BM + self.ovalo, y_BM + self.ovalo, fill="green")

			
			self.kalman_NCV.x = x_backup_NCV
			self.kalman_BM.x = x_backup_BM
			self.kalman_NCV.P = p_backup_NCV
			self.kalman_BM.P = p_backup_BM
		self.samples += 1

	def calculate_rmse(self):
		errors_NCV = []
		errors_BM = []

		for i in range(len(self.history)):
			true_x, true_y = self.history[i]
			ncv_x, ncv_y = self.predictions_NCV[i]
			bm_x, bm_y = self.predictions_BM[i]

			# Calculate Euclidean distance for each prediction
			error_NCV = np.sqrt((true_x - ncv_x)**2 + (true_y - ncv_y)**2)
			error_BM = np.sqrt((true_x - bm_x)**2 + (true_y - bm_y)**2)
			
			errors_NCV.append(error_NCV)
			errors_BM.append(error_BM)

		rmse_NCV = np.sqrt(np.mean(np.array(errors_NCV)**2))
		rmse_BM = np.sqrt(np.mean(np.array(errors_BM)**2))

		return rmse_NCV, rmse_BM

class Kalman:
	def __init__(self, tipo, x_init):
		self.tipo = tipo

		# Init the state
		self.x = np.array([
			[x_init[0]],
			[x_init[1]],
			[x_init[2]],
			[x_init[3]]
			])
		self.P = np.eye(4)

		# Measure equation
		self.H = np.array([[1, 0, 0, 0],
						   [0, 1, 0, 0]])
		# Covariance matrices
		self.R = np.eye(2) * 1.0
		self.Q = np.eye(4) * 0.000001

	def generateF(self, dt):
		if self.tipo == "NCV":
			F = np.array(
				[[1, 0, dt, 0],
				 [0, 1, 0, dt],
				 [0, 0, 1, 0],
				 [0, 0, 0, 1]])  # Near Constant Velocity
		elif self.tipo == "BM":
			F = np.eye(4)
		return F
	def predict(self, dt):
		F = self.generateF(dt)
		# Calcular el prior
		self.x = F @ self.x
		self.P = F @ self.P @ F.T + self.Q

	def update(self, z):

		y = z - self.H @ self.x

		# Kalman gain
		s = self.H @ self.P @ self.H.T + self.R
		k = self.P @ self.H.T @ np.linalg.inv(s)

		# posterior
		self.x = self.x + k @ y

		I = np.eye(4)
		self.P = (I - k @ self.H) @ self.P


if __name__ == "__main__":
	root = tk.Tk()
	
	arena = Arena(root, 800, 600)

	def on_close():
		rmse_NCV, rmse_BM = arena.calculate_rmse()
		print(f"RMSE NCV: {rmse_NCV}")
		print(f"RMSE BM: {rmse_BM}")
		root.quit()

	root.protocol("WM_DELETE_WINDOW", on_close)
	root.mainloop()

############################################################################
### Conclusiones al cambiar valores de la covarianza de proceso y medida

# Cuando Q es baja: El filtro confía mucho en el modelo, siguiendo su trayectoria (suaviza más).
# Cuando Q es alta: El filtro desconfía del modelo, reaccionando más a las mediciones (menos suavizado).
# Cuando R es baja: El filtro confía mucho en las medidas, siguiéndolas de cerca (muy ruidoso).
# Cuando R es alta: El filtro desconfía de las medidas, suavizando la salida (prioriza el modelo).
