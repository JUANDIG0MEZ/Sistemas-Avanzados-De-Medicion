import matplotlib.pyplot as plt



import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)


plt.plot(x, y)
plt.title('Sine Wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid()
plt.savefig('./imagenes/sine_wave.png')
plt.show()
# save the figure
