import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 2
X0 = 1
t1 = 20

# Function to solve the differential equation numerically using Euler's method
def numerical_Euler_solver(x0=X0, l=LAMBDA, dt=0.1, t1=20):
    n_steps = int(t1/dt)
    sol = np.array([(x0 * ((1 - l * dt)**n)) for n in range(n_steps)])
    times = np.cumsum(np.ones(n_steps) * dt)
    times[0] = 0
    return (times, sol)

# def analytical_solution(x0, l=LAMBDA, t=0):
#     return x0 * np.exp(-l*t)

analytical_solution = lambda t, l: X0 * np.exp(-l*t)
# err = lambda x0, t, l, dt: x0 * np.exp(-l*t) - x0 * ((1 - l * dt)**t)
def q_1():
    for dt in [0.4, 0.93, 1.02]:
        times, sol = numerical_Euler_solver(dt=dt)
        plt.scatter(times, sol, label=f"dt={dt}")
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.title(f'X as function of time.\n' + r'$\lambda$' + f'={LAMBDA}')
    plt.legend()
    plt.show()




if __name__ == "__main__":
    q_1()