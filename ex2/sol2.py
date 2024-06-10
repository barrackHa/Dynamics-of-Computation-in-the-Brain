"""
Solution for exercise 1.
Dynamcs Of Computation In The Brain - 76908
By: Barak H.
May 2024
"""
import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 2
X0 = 1
t1 = 20

def numerical_Euler_solver(x0=X0, l=LAMBDA, dt=0.1, t1=20):
    """Numerical solver using Euler's methode for the ODE dx/dt = -l*x"""
    n_steps = int(t1/dt)
    # Vectorized way of writing: sol = np.array([(x0 * ((1 - l * dt)**n)) for n in range(n_steps)])
    sol = x0 * (
        (np.ones(n_steps) * (1 - l * dt)) ** (np.arange(0, n_steps))
    )
    times = np.arange(0, sol.size*dt, dt)
    return (times, sol)

def q_1():
    """Question 1: Plot the solution for different dt values."""
    for dt in [0.4, 0.93, 1.02]:
        times, sol = numerical_Euler_solver(dt=dt)
        plt.scatter(times, sol, label=f"dt={dt}")
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.title(f'X as function of time.\n' + r'$\lambda$' + f'={LAMBDA}')
    plt.legend()
    plt.show()

# Lambda function for the analytical solution, i.e. x(t) = x0 * exp(-l*t)
analytical_solution = lambda t: X0 * np.exp(-1*LAMBDA*t)

def q_2():
    """Question 2: Plot the error as a function of time."""
    times, sol = numerical_Euler_solver(dt=0.2)
    errors = np.abs(analytical_solution(times) - sol)
    plt.plot(times, errors)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Error')
    plt.title(f'Error as function of time.\n' + r'$\lambda$' + f'={LAMBDA}')
    plt.show()


if __name__ == "__main__":
    # q_1()
    # q_2()
    # fig, ax = plt.subplots()
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

    # Define the system of ODEs
    def linear_system(t, h, M, h_0):
        """
        The system of ODEs for the linear system of equations.
        h = [h_E, h_I]
        h_0 = [h_E0, h_I0] is the external input.
        """
        return (M @ h) + h_0

    # Define the 2x2 matrix A
    M = np.array([[-1, 1],
                [0, -1]])
    
    J_EE, J_EI = 2, 2
    M_1 = np.array([
        [(J_EE - 1), (-1 * J_EI)],
        [1, -1]
    ])

    M_3 = np.array([
        [-1, -1*J_EI],
        [0, -1]
    ])

    # 4. (h_E < 0) && (0 < h_I)
    # h_ext = [h_E^0, h_I^0]
    h_0 = np.array([-10, -10])
    # Initial conditions
    y0 = [6, 2]
    NPOINTS = 1000
    # Time span for the simulation
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], NPOINTS)

    # Solve the system of ODEs
    solution = solve_ivp(linear_system, t_span, y0, args=(M_1, h_0), t_eval=t_eval)

    # Plot the solution
    fig, ax = plt.subplots()
    
    cmap = plt.cm.get_cmap('Blues', NPOINTS)
    colorpoints = np.linspace(0, 1, NPOINTS)
    sc = ax.scatter(solution.y[0], solution.y[1], c=colorpoints, cmap=cmap)
    # Add a color bar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Time')

    ax.set_xlabel('y_0')
    ax.set_ylabel('y_1')
    # ax.legend()
    # ax.suptitle('Solution of the Coupled 2D Linear System')
    ax.grid()
    # plt.show()

    fig_fft, (r_vs_t, psd) = plt.subplots(2)
    r_vs_t.plot(solution.t, solution.y[0], label=r'$r_{E}$')

    # Run numpy fft on solution.y[0]
    print(solution.y[0].shape)
    fft = np.fft.fft(solution.y[0])
    print(fft.shape)
    # freqs = np.fft.fftfreq(len(fft), d=1/NPOINTS)
    freqs = np.fft.fftfreq(len(fft), d=1/10)
    print(freqs.shape)
    # Plot the power spectrum
    psd.plot(freqs[:len(fft)//2], np.abs(fft)[:len(fft)//2])
    # psd.plot(np.abs(fft))
    # psd.set_xlabel('Frequency')
    # psd.set_ylabel('Power')
    # psd.legend()
    # psd.grid()


    plt.show()