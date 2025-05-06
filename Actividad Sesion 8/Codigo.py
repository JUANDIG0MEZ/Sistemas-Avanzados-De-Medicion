# A Kalman filter premier by German Holguin
#% Purdue Robot Vision Lab 2006
#% The very first simple linear scalar case.

#% In this example, the mean value chenges over time.

#% The first value will be 1.5
#% The second value will be 0.5

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand

# NUMBERS OF SAMPLES PER VALUE
N_SAMPLES = 1000
# The main assumption of the Kalman Filter is that both process and measurement noises are Gaussian
Samples1 = np.random.normal(1.5, 0.01, int(N_SAMPLES/2))
Samples2 = np.random.normal(0.5, 0.01, int(N_SAMPLES/2))
Samples = np.concatenate((Samples1, Samples2), axis=0)


# Mean and Standard deviation for the generated samples are:
Sm = np.mean(Samples)  # This value is used only to plot the ground truth
Sdv = np.std(Samples)  # This value can be use to model the measurement noise

# The State Space Model
# Since the state is directly observed, and the voltage we are measuring is DC, we have:
# x[k+1] = x[k] + w[k]
#   y[k] = x[k] + v[k]

# Tunning parameters
# Q: Variance in the process noise w[k]
Q = 1e-1
# R: Variance in the measurement noise v[k]
R = Sdv**2   # <<< Let's say we have been observing it for a while and we have a very good estimate of it.

# Initial values

X = np.zeros(N_SAMPLES)
P = np.zeros(N_SAMPLES)
Xposterior = 0
Pposterior = 1

F = 1  # State transition matrix
H = 1  # Measurement matrix


for i in range(N_SAMPLES):
		
	# Sample arrives
	z = Samples[i]
	# PREDICT
	Xprior = F*Xposterior
	Pprior = F*Pposterior + Q

	# UPDATE
	K = (Pprior*H)/(H*Pprior + R)  # Kalman Gain
	Xposterior = Xprior + K*(z - H*Xprior)  # Update the state estimate with the measurement
	Pposterior = (np.eye(np.size(K*H)) - K*H)*Pprior  # Update the error covariance

	# store result
	X[i] = Xposterior
	P[i] = Pposterior


# PLOTTING THE RESULTS

# The Kalman Filter output is stored in X
plt.figure(figsize=(10, 5))
plt.plot(Samples, '+r', label='Mediciones')
plt.plot(X, '*b', label='Estimación Kalman')
plt.plot(np.concatenate([np.ones(int(N_SAMPLES/2))*1.5, np.ones(int(N_SAMPLES/2))*0.5]), 'g', label='Valor Real')
plt.legend()
plt.xlabel('Tiempo (iteración)')
plt.ylabel('Valor')
plt.title('Filtro de Kalman - Caso Escalar Simple')
plt.grid(True)
plt.show()

# figure(1)
# hold off
# # Samples are plotted in RED
# plot(Samples,'+r');
# hold on
# axis([0 N_SAMPLES 0 2]);
# # Kalman Filter Output is plotted in BLUE
# plot(X,'*b');
# # Ground Truth is plotted in GREEN
# plot([1.5*ones(1,N_SAMPLES/2) 0.5*ones(1,N_SAMPLES/2)],'g');