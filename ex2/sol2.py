"""
Solution for exercise 2.
Dynamcs Of Computation In The Brain - 76908
By: Barak H.
June 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq

def linear_system(t, h, M, h_0):
    """
    The system of ODEs for the linear system of equations.
    h = [h_E, h_I]
    h_0 = [h_E_0, h_I_0] is the external input.
    """
    return (M @ h) + h_0

M = np.array([
    [-1, 1],
    [0, -1]]
)

def M_1(J_EE=2, J_EI=2):
    """
    Returns the matrix M_1 for the system of ODEs.
    The conditions are (0 < h_E) && (h_I < 0) from Q2.1.1.1
    """
    return np.array([
        [(J_EE - 1), (-1 * J_EI)],
        [1, -1]
    ])

def M_3(J_EI=2):
    """
    Returns the matrix M_3 for the system of ODEs.
    The conditions are (h_E <= 0) && (0 < h_I)
    """
    return np.array([
        [-1, (-1 * J_EI)],
        [-1, -1]
    ])

def get_simulation_results(Mat, external_input, initial_conditions, time_span, points_num):
    t_eval = np.linspace(time_span[0], time_span[1], points_num)
    solution = solve_ivp(
        linear_system, time_span, initial_conditions, 
        args=(Mat, external_input), t_eval=t_eval
    )
    return solution

def q_2_plot_helper(ax, h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, fig=None):

    # Verify the fixed point conditions
    A = (0 < 1 - J_EE + J_EI) 
    B = (0 < h_E_0 - (h_I_0 * J_EI)) 
    A_n_B = A and B
    notA_n_notB = (not A) and (not B)
    if (not A_n_B) and (not notA_n_notB):
        err = "The fixed point conditions are not met.\n"
        err += f'1 - J_EE + J_EI = 1 - {J_EE} + {J_EI} = {1 - J_EE + J_EI}' + '\n'
        err += f'h_E_0 - (J_EI * h_I_0) = {h_E_0} - ({J_EI} * {h_I_0}) = {h_E_0 - (J_EI * h_I_0)}'
        # raise ValueError(f'{err}')
        print(f'{err}')
    
    h_E_star = (h_E_0 - (J_EI * h_I_0)) / (1 - J_EE + J_EI)
    h_I_star = h_E_star + h_I_0

    h_0 = np.array([h_E_0, h_I_0])

    print(f'Fixed point: ({h_E_star}, {h_I_star})')
    dx = np.abs(h_E_star - 0.5) 
    dy = np.abs(h_I_star - 0.5)
    # Solve the system of ODEs
    for a,b in [[1,1], [-1,1], [1,-1], [-1,-1]]:
        y0 = [(h_E_star + a*dx), (h_I_star + b*dy)]
        solution = get_simulation_results(M_1(J_EE,J_EI), h_0, y0, t_span, NPOINTS)
        # Plot the solution
        ax.plot(solution.y[0], solution.y[1], label=f'Starting At ({y0[0]:.2f},{y0[1]:.2f})')
        sc = ax.scatter(solution.y[0], solution.y[1], c=solution.t, cmap='viridis', alpha=0.7)
        
    sc = ax.scatter(solution.y[0], solution.y[1], c=solution.t, cmap='viridis', alpha=0.7)
    
    if fig is not None:
        # Add a color bar
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label('Time')

        ttl = "Solution of the Coupled 2D Linear System\n"
        ttl += r'$h_{E}^{0}$=' + f'{h_E_0}, ' + r'$h_{I}^{0}$=' + f'{h_I_0} '
        ttl += f"Starting at {t_span[0]} to {t_span[1]}" 
        fig.suptitle(ttl)

    # Plot the fixed point
    ax.scatter(
        h_E_star, h_I_star, color='red', s=200, linewidths=1, edgecolors='black', 
        marker='P', label=f'FP=({h_E_star:.2f},{h_I_star:.2f})'
    )

    ax.set_xlabel(r'$r_{E}$')
    ax.set_ylabel(r'$r_{I}$', rotation=0)
    ax.legend()
    ttl = r'$J_{EJ}$=' + f'{J_EE}, ' + r'$J_{EI}$=' + f'{J_EI}'
    ax.set_title(ttl)
    ax.grid()
    ax.set_facecolor('lightgray')
    # ax.set_xlim([-30, 30])
    # ax.set_ylim([-30, 30])
    return fig, ax

def q_2_2_1_real_stable():
    fig, axs = plt.subplots(1,2, figsize=(14,8))
    h_E_0, h_I_0 = np.array([1, 1])
    t_span, NPOINTS = (0, 15), 1000

    J_EE, J_EI = 1, 0.2 # Stable 1
    q_2_plot_helper(axs[0], h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, fig=None)
    
    J_EE_2, J_EI_2 = 1.1, 0.25 # Stable 2
    q_2_plot_helper(axs[1], h_E_0, h_I_0, J_EE_2, J_EI_2, t_span, NPOINTS, fig=fig)
    
    return {
        'fig': fig,
        'axs': axs,
        't_span': t_span,
        'NPOINTS': NPOINTS,
        'h_0': np.array([h_E_0, h_I_0]),
        'initial_conditions': np.array([[J_EE, J_EI], [J_EE_2, J_EI_2]]),
    }

def q_2_2_1_real_unstable():
    fig, axs = plt.subplots(1,2, figsize=(14,8))
    h_E_0, h_I_0 = np.array([1, 1])
    t_span, NPOINTS = (0, 5), 100

    J_EE, J_EI = 2.5, 1.1 # unStable 1
    q_2_plot_helper(axs[0], h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, fig=None)
    
    J_EE_2, J_EI_2 = 3, 1.75 # unStable 2
    q_2_plot_helper(axs[1], h_E_0, h_I_0, J_EE_2, J_EI_2, t_span, NPOINTS, fig=fig)
    
    return {
        'fig': fig,
        'axs': axs,
        't_span': t_span,
        'NPOINTS': NPOINTS,
        'h_0': np.array([h_E_0, h_I_0]),
        'initial_conditions': np.array([[J_EE, J_EI], [J_EE_2, J_EI_2]]),
    }

def q_2():
    q_2_2_1_real_stable()
    q_2_2_1_real_unstable()
    plt.show()
    exit()
    # 4. (h_E < 0) && (0 < h_I)
    # h_ext = [h_E^0, h_I^0]
    fig, ax = plt.subplots()

    h_E_0, h_I_0 = h_0 = np.array([2, 1])
    # J_EE, J_EI = 1, 0.2 # Stable
    # J_EE, J_EI = 4, 3.1 # Unstable
    # J_EE, J_EI = 1, 1 # Stable spiral
    J_EE, J_EI = 3, 3 # Unstable spiral

    # Initial conditions
    # y0 = [1, 1]
    NPOINTS = 1000
    # Time span for the simulation
    t_span = (0, 10)
    # t_eval = np.linspace(t_span[0], t_span[1], NPOINTS)

    h_E_star = (h_E_0 - (J_EI * h_I_0)) / (1 - J_EE + J_EI)
    h_I_star = h_E_star + h_I_0

    print(f'Fixed point: ({h_E_star}, {h_I_star})')
    dx = np.abs(h_E_star - 0.5) 
    dy = np.abs(h_I_star - 0.5)
    # Solve the system of ODEs
    # solution = solve_ivp(linear_system, t_span, y0, args=(M_1, h_0), t_eval=t_eval)
    # for y0 in [[1,1], [-1,2], [2,-1], [-2,-2]]:
    # for y0 in [[5.1,0.1], [19,0.1], [0.1,20], [20,20.1]]:
    for a,b in [[1,1], [-1,1], [1,-1], [-1,-1]]:
        y0 = [(h_E_star + a*dx), (h_I_star + b*dy)]
        # Cut off y0 elements after 2 decimal points
        y0 = [round(i, 2) for i in y0]
        solution = get_simulation_results(M_1(J_EE,J_EI), h_0, y0, t_span, NPOINTS)
        # Plot the solution
        ax.plot(solution.y[0], solution.y[1], label=f'y_0={y0}', linewidth=1, alpha=1)
        sc = ax.scatter(solution.y[0], solution.y[1], c=solution.t, cmap='viridis', alpha=0.7)
        # solution = get_simulation_results(M_1(1,2), h_0, y0, t_span, NPOINTS)
    # Plot the solution
    
    
    # cmap = plt.cm.get_cmap('Blues', NPOINTS)
    # colorpoints = np.linspace(0, 1, NPOINTS)
    # sc = ax.scatter(solution.y[0], solution.y[1], c=colorpoints, cmap=cmap)
    sc = ax.scatter(solution.y[0], solution.y[1], c=solution.t, cmap='viridis', alpha=0.7)
    # Add a color bar
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Time')

    ax.scatter(
        h_E_star, h_I_star, color='red', s=200, linewidths=1, edgecolors='black', 
        marker='P', label=f'FP=({h_E_star:.2f},{h_I_star:.2f})'
    )

    ax.set_xlabel(r'$r_{E}$')
    ax.set_ylabel(r'$r_{I}$', rotation=0)
    ax.legend()
    # ax.suptitle('Solution of the Coupled 2D Linear System')
    ax.grid()
    #set ax limits
    ax.set_facecolor('lightgray')
    # ax.set_xlim([-30, 30])
    # ax.set_ylim([-30, 30])
    plt.show()
    
if __name__ == "__main__":

    q_2()
    J_EE = J_EI = 2
    t_span, NPOINTS = (0, 100), 100
    h_0 = np.array([6, 2])
    y0 = [-10, -10]
    solution = get_simulation_results(M_1(J_EE,J_EI), h_0, y0, t_span, NPOINTS)

    t = np.linspace(0, 100, 100)
    x = np.sin(t)
    fig_fft, (r_vs_t, psd) = plt.subplots(2)
    # r_vs_t.plot(solution.t, solution.y[0], label=r'$r_{E}$')
    r_vs_t.plot(t, x, label=r'x')

    # Run numpy fft on solution.y[0]
    print(solution.y[0].shape)
    # fft = np.fft.fft(solution.y[0])
    fft = np.fft.fft(x)
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