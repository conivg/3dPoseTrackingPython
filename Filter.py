import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import lfilter
mu, sigma = 0,500

x = np.arange(1,100,0.1) #x axis

z = np.random.normal(mu, sigma, len(x))#noise

y = x**2 + z #data

#plt.plot(x,y,linewidth=2,linestyle="-", c="b") #it includes some noise

n= 100
b = [1.0/n]*n
a = 1
yy = lfilter(b,a,y)
print(yy)
print(yy[-1])
plt.plot(x,yy,linewidth=2,linestyle="-", c="b") #it includes some noise
plt.show()
