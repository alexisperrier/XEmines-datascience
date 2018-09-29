import matplotlib.pyplot as plt
import numpy as np

max_iters   = 10000
precision   = 0.0001

gammas       = np.linspace(0.001, 0.01, 10)

fct = lambda x: 4 * x**3 - 9 * x**2

def gradient(gamma):
    iters       = 0
    cur_x       = 6
    previous_step_size = 1
    x = []

    while (previous_step_size > precision) & (iters < max_iters):
        x.append(cur_x)
        prev_x = cur_x
        cur_x -= gamma * fct(prev_x)
        previous_step_size = abs(cur_x - prev_x)
        iters+=1

    print("Gamma {}  min {:.4f} f(min) {:.4f}".format(gamma, cur_x, fct(cur_x)))
    return x

gamma = gammas[0]

fig, ax = plt.subplots(1,1)
for gamma in gammas:
    x = gradient(gamma)
    plt.plot(x, label = gamma)
    plt.legend()
    plt.show()

plt.legend()
plt.show()
