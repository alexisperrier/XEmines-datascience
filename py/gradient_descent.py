
# gamma       = 0.01 # step size multiplier
max_iters   = 10000 # maximum number of iterations

precision   = 0.00001


# dérivée de la fonction à minimiser

gammas       = [0.01, 0.025, 0.05, 0.075, 0.1] # learning_rate

fct = lambda x: 4 * x**3 - 9 * x**2

x = {}

for gamma in gammas:
    print(gamma)
    iters       = 0 #iteration counter
    cur_x       = 6 # The algorithm starts at x=6
    previous_step_size = 1
    x[gamma] = []
    while (previous_step_size > precision) & (iters < max_iters) & (previous_step_size < 10):
        x[gamma].append(cur_x)
        prev_x = cur_x
        cur_x -= gamma * fct(prev_x)
        print(cur_x, previous_step_size)
        previous_step_size = abs(cur_x - prev_x)
        iters+=1

    print("Gamma {}  min {:.4f} f(min) {:.4f}".format(gamma, cur_x, fct(cur_x)))
