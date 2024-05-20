import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 2
X0 = 1
t1 = 20

# Function to solve the differential equation numerically using Euler's method
def numerical_Euler_solver(x0=X0, l=LAMBDA, dt=0.1, t1=20):
    n_steps = int(t1/dt)
    sol = np.array([(x0 * ((1 - l * dt)**n)) for n in range(n_steps)])
    times = np.arange(0, sol.size*dt, dt)
    return (times, sol)

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

analytical_solution = lambda t: X0 * np.exp(-1*LAMBDA*t)

def q_2():
    times, sol = numerical_Euler_solver(dt=0.2)
    errors = np.abs(analytical_solution(times) - sol)
    plt.plot(times, errors)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(f'Error as function of time.\n' + r'$\lambda$' + f'={LAMBDA}')
    plt.show()


if __name__ == "__main__":
    q_1()
    q_2()