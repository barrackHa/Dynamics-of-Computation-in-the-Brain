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

def q_2_plot_helper(ax, h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, fig=None, e_type=''):

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

    print(f'Fixed point: ({h_E_star:.2f}, {h_I_star:.2f})')
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
        # And a title
        type_str = f'Of Type: {e_type}' if e_type!='' else ''
        ttl = "Solution of the Coupled 2D Linear System " + type_str + "\n"
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
    ttl = r'$J_{EE}$=' + f'{J_EE}, ' + r'$J_{EI}$=' + f'{J_EI}'
    ax.set_title(ttl)
    ax.grid()
    ax.set_facecolor('lightgray')
    # ax.set_xlim([-30, 30])
    # ax.set_ylim([-30, 30])
    return fig, ax

def plot_helper_wrapper(Js_array, h_E_0=2, h_I_0=1, t_span=(0, 15), NPOINTS=1000, type_str=''):
    fig, axs = plt.subplots(1, Js_array.shape[0], figsize=(14,8))

    if Js_array.shape[0] == 1:
        axs = np.array([axs])
    
    for i, (J_EE, J_EI) in enumerate(Js_array):
        fig_obj = None if (i<Js_array.shape[0]-1) else fig
        q_2_plot_helper(axs[i], h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, fig=fig_obj, e_type=type_str)
    
    return {
        'fig': fig,
        'axs': axs,
        't_span': t_span,
        'NPOINTS': NPOINTS,
        'h_0': np.array([h_E_0, h_I_0]),
        'initial_conditions': np.array(Js_array),
    }

def q_2_2_1_real_stable():
    J_EE, J_EI = 1, 0.2 # Stable 1
    J_EE_2, J_EI_2 = 1.1, 0.25 # Stable 2
    return plot_helper_wrapper(np.array([
        [J_EE, J_EI],
        [J_EE_2, J_EI_2]
    ]), type_str='Real Stable')
    

def q_2_2_1_real_unstable():
    t_span, NPOINTS = (0, 5), 100
    J_EE, J_EI = 4, 2.2 # unStable 1
    J_EE_2, J_EI_2 = 4.1, 2.3 # unStable 2
    
    return  plot_helper_wrapper(np.array([
        [J_EE, J_EI],
        [J_EE_2, J_EI_2]
    ]), t_span=t_span, NPOINTS=NPOINTS, type_str='Real UnStable')

def q_2_2_1_complex_stable():
    J_EE, J_EI = 0.6, 0.9 # Complex Stable 1
    J_EE_2, J_EI_2 = 1.5, 0.85 # Complex Stable 2
    return plot_helper_wrapper(np.array([
        [J_EE, J_EI],
        [J_EE_2, J_EI_2]
    ]), type_str='Complex Stable')

def q_2_2_1_complex_unstable():
    J_EE, J_EI = 2.5, 1.75 # Complex unStable 1
    J_EE_2, J_EI_2 = 2.6, 1.9 # Complex unStable 2
    return plot_helper_wrapper(np.array([
        [J_EE, J_EI],
        [J_EE_2, J_EI_2]
    ]), type_str='Complex UnStable')

def q_2():
    return [
        q_2_2_1_real_stable(),
        q_2_2_1_real_unstable(),
        q_2_2_1_complex_stable(),
        q_2_2_1_complex_unstable()
    ]

def q_3_1(J_EE=2, J_EI=2, t_span=(0, 100), NPOINTS=100, h_0=np.array([2, 1])):    
    y0 = [1, 1]
    
    solution = get_simulation_results(M_1(J_EE,J_EI), h_0, y0, t_span, NPOINTS)
    fig, ax = plt.subplots()
    fig.suptitle(f'Q3.1: J_EE={J_EE}, J_EI={J_EI}, y0={y0}, h0={h_0}')
    ax.plot(solution.t, solution.y[0], label=r'$r_{E}$', linewidth=2)
    ax.set_xlabel('Time [Seconds]')
    ax.set_ylabel(r'$r_{E}$', rotation=0)
    ax.set_facecolor('lightgray')
    ax.grid()

    return fig, ax, solution

def q_3_2_1(sol=None, t_span=(0, 100), NPOINTS=100):
    if sol is None:
        _, _, sol = q_3_1(t_span=t_span, NPOINTS=NPOINTS)
    
    # Run scipy fft on solution.y[0]
    r_E_fft = fft(sol.y[0])
    freqs = fftfreq(len(r_E_fft), d=1)

    # Plot the power spectrum
    fig_fft, psd = plt.subplots()
    psd.plot(freqs[0<=freqs], np.abs(r_E_fft)[0<=freqs])
    psd.set_xlabel('Frequency [Hz]')
    psd.set_ylabel('Power')
    psd.set_title(r'Power Spectrum Of $r_{E}$')
    max_freq = freqs[np.abs(r_E_fft).argmax()]
    psd.vlines(
        max_freq, ymin=0, ymax=np.abs(r_E_fft).max(), 
        color='orange', linestyle='--', linewidth=2, 
        label=f'Peak Frequency at ~ {max_freq:.2f}[Hz]'
    )
    psd.grid()
    psd.legend()
    psd.set_facecolor('lightgray')

def q_3_2_4():
    
    # plt.cla()
    # plt.clf()

    plt, (sig, psd) = plt.subplots(2)
    for J_EE, J_EI in [[1, 0.2], [4, 2.2], [0.6, 0.9], [2.5, 1.75]]:
        _, _, sol = q_3_1()
        sig.plot(sol.t, sol.y[0], label=f'J_EE={J_EE}, J_EI={J_EI}')
    


def q_3():
    # t_span, NPOINTS = (0, 100), 100
    # _, _, sol_q_3_1 = q_3_1()
    # q_3_2_1()
    q_3_2_4()

if __name__ == "__main__":

    # q_2()
    # plt.show()

    q_3()
    plt.show()
    exit()
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