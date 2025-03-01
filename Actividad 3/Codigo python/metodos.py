import math
import random

def boxMuller(mu= 0, sigma=1):
    u1 = random.random()
    u2 = random.random()

    z0 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    z1 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)

    z0 = sigma * z0 + mu
    z1 = sigma * z1 + mu

    return z0, z1




def rungeKutta4():
    print("Metodod de runge-kutta 4")
