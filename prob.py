import numpy as np
import matplotlib.pyplot as plt

n = 10000

A = np.random.uniform(0, 100, size=n)
B = np.random.uniform(-100, 0, size=n)
w = 2
t_1 = np.linspace(-100, 100, num=n)
t_2 = np.linspace(-100, 100, num=n)

X_1 = np.cos(A * np.cos(w*t_1) + B * np.sin(w*t_1))
X_2 = np.cos(A * np.cos(w*t_2) + B * np.sin(w*t_2))
plt.hist(X_1, density=True, color='red', bins=int(n/100))
plt.hist(X_2, density=True, color='cyan', bins=int(n/100))

print("<X(t_1)X(t_2)> =", np.average(X_1*X_2))
print("<X(t_1)> =", np.average(X_1))

def F_1(x_1):
    return np.count_nonzero(X_1 < x_1)/len(X_1)

xvals = np.linspace(min(X_1), max(X_1), num=n)
F1_arr = np.array([F_1(x) for x in xvals])
plt.plot(xvals, F1_arr, color='blue')


f = 1/(np.pi * np.sqrt(1-t_1**2))
plt.plot(t_1, f)
plt.show()