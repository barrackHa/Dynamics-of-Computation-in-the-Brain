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

def excitatory_inhibitor_system(t, h, J_EE=2, J_EI=1, h_0=[2,1]):
    """
    The system of ODEs for the excitatory-inhibitory system.
    h = [h_E, h_I]
    """
    # simple Relu function 
    h_E_p = h[0] * (h[0] > 0)
    h_I_p = h[1] * (h[1] > 0)

    h_E_0, h_I_0 = h_0

    dh_E = (-1 * h_E_p) + (J_EE * h_E_p) - (J_EI * h_I_p) + h_E_0
    dh_I = (-1 * h_I_p) + h_E_p + h_I_0
    
    return np.array([dh_E, dh_I])

def simulate_excitatory_inhibitory_system(
    J_EE=2, J_EI=1, h_0=[2,1], initial_conditions=[1,1], time_span=(0, 100), points_num=100
):
    """
    Simulate the excitatory-inhibitory system.
    """
    t_eval = np.linspace(time_span[0], time_span[1], points_num)
    solution = solve_ivp(
        excitatory_inhibitor_system, time_span, initial_conditions, 
        args=(J_EE, J_EI, h_0), t_eval=t_eval
    )
    return solution

def get_simulation_results(Mat, external_input, initial_conditions, time_span, points_num):
    t_eval = np.linspace(time_span[0], time_span[1], points_num)
    solution = solve_ivp(
        linear_system, time_span, initial_conditions, 
        args=(Mat, external_input), t_eval=t_eval
    )
    return solution

def q_2_plot_helper(
        ax, h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, 
        fig=None, e_type='', use_linear_system=False
    ):

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

        if use_linear_system:
            solution = get_simulation_results(M_1(J_EE,J_EI), h_0, y0, t_span, NPOINTS)
        else: 
            solution = simulate_excitatory_inhibitory_system(J_EE, J_EI, h_0, y0, t_span, NPOINTS)
        
        # Plot the solution
        ax.plot(solution.y[0], solution.y[1], label=f'Starting At ({y0[0]:.2f},{y0[1]:.2f})')
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

def plot_helper_wrapper(Js_array, h_E_0=2, h_I_0=1, t_span=(0, 15), 
        NPOINTS=1000, type_str='', use_linear_system=False):
    fig, axs = plt.subplots(1, Js_array.shape[0], figsize=(14,8))

    if Js_array.shape[0] == 1:
        axs = np.array([axs])
    
    for i, (J_EE, J_EI) in enumerate(Js_array):
        fig_obj = None if (i<Js_array.shape[0]-1) else fig
        q_2_plot_helper(
            axs[i], h_E_0, h_I_0, J_EE, J_EI, t_span, NPOINTS, 
            fig=fig_obj, e_type=type_str, use_linear_system=use_linear_system
        )
    
    return {
        'fig': fig,
        'axs': axs,
        't_span': t_span,
        'NPOINTS': NPOINTS,
        'h_0': np.array([h_E_0, h_I_0]),
        'J_values': np.array(Js_array),
    }

def q_2_2_1_real_stable(use_linear_system=False):
    J_EE, J_EI = 1, 0.2 # Stable 1
    J_EE_2, J_EI_2 = 1.1, 0.25 # Stable 2
    return plot_helper_wrapper(np.array([
            [J_EE, J_EI],
            [J_EE_2, J_EI_2]
        ]), type_str='Real Stable',
        use_linear_system=use_linear_system
    )
    

def q_2_2_1_real_unstable(use_linear_system=False):
    t_span, NPOINTS = (0, 5), 100
    J_EE, J_EI = 4, 2.2 # unStable 1
    J_EE_2, J_EI_2 = 4.1, 2.3 # unStable 2
    
    return  plot_helper_wrapper(np.array([
            [J_EE, J_EI],
            [J_EE_2, J_EI_2]
        ]), t_span=t_span, NPOINTS=NPOINTS, type_str='Real UnStable',
        use_linear_system=use_linear_system
    )

def q_2_2_1_complex_stable(use_linear_system=False):
    J_EE, J_EI = 0.6, 0.9 # Complex Stable 1
    J_EE_2, J_EI_2 = 1.5, 0.85 # Complex Stable 2
    return plot_helper_wrapper(np.array([
            [J_EE, J_EI],
            [J_EE_2, J_EI_2]
        ]), type_str='Complex Stable', 
        use_linear_system=use_linear_system
    )

def q_2_2_1_complex_unstable(use_linear_system=False):
    J_EE, J_EI = 2.1, 1.25 # Complex unStable 1
    # J_EE, J_EI = 2.5, 1.75 # Complex unStable 1
    J_EE_2, J_EI_2 = 2.6, 1.9 # Complex unStable 2
    return plot_helper_wrapper(np.array([
            [J_EE, J_EI],
            [J_EE_2, J_EI_2]
        ]), type_str='Complex UnStable', t_span=(0, 15), NPOINTS=1000,
        use_linear_system=use_linear_system
    )

def q_2_bonus():
    J_EE, J_EI = 2, 4
    dict = plot_helper_wrapper(np.array([
            [J_EE, J_EI],
        ]), type_str='Complex UnStable', t_span=(0, 20), NPOINTS=1000,
        h_E_0=36, h_I_0=1, use_linear_system=True
    )
    
    h_0 = dict['h_0']
    y0 = [20.83,20.83]
    t_span = dict['t_span']
    NPOINTS = dict['NPOINTS']
    
    solution = simulate_excitatory_inhibitory_system(J_EE, J_EI, h_0, y0, t_span, NPOINTS)
    sys_fig, sys_ax = plt.subplots(figsize=(10,6))
    sys_ax.plot(solution.y[0], solution.y[1], label=f'Starting At ({y0[0]:.2f},{y0[1]:.2f})')
    sc = sys_ax.scatter(solution.y[0], solution.y[1], c=solution.t, cmap='viridis', alpha=0.7)
    cbar = sys_fig.colorbar(sc, ax=sys_ax)
    cbar.set_label('Time')
    ttl = "Solution of the 2D Original System - Boarder Of Instability\n"
    ttl += r'$h_{E}^{0}$=' + f'{h_0[0]}, ' + r'$h_{I}^{0}$=' + f'{h_0[1]} '
    ttl += f"Starting at {t_span[0]} to {t_span[1]}" 
    sys_fig.suptitle(ttl)
    sys_ax.set_xlabel(r'$r_{E}$')
    sys_ax.set_ylabel(r'$r_{I}$', rotation=0)
    sys_ax.legend()
    sys_ax.grid()
    sys_ax.set_facecolor('lightgray')
    ttl = r'$J_{EE}$=' + f'{J_EE}, ' + r'$J_{EI}$=' + f'{J_EI}'
    sys_ax.set_title(ttl)

    max_freq = q_3_2_1_fft_helper(solution)
    fig, ax = plt.subplots()
    ax.plot(solution.t, solution.y[0], label=r'$r_{E}$', linewidth=2)
    ax.set_title(f'Max Freq Using FFT Is: {max_freq:.2f}[Hz]')
    ax.set_xlabel('Time [Seconds]')
    ax.set_ylabel(r'$r_{E}$', rotation=0)
    ax.grid()
    ax.set_facecolor('lightgray')
    fig.suptitle('Q2 Bonus - Simulation Of The Excitatory-Inhibitory System With J_EE=2, J_EI=4\n')
    return fig, ax
        

def q_3_1_1(J_EE=2, J_EI=2, t_span=(0, 100), NPOINTS=100, h_0=np.array([32, 1]), y0=[1,1]):    
    # solution = get_simulation_results(M_1(J_EE,J_EI), h_0, y0, t_span, NPOINTS)
    solution = simulate_excitatory_inhibitory_system(J_EE, J_EI, h_0, y0, t_span, NPOINTS)
    fig, ax = plt.subplots()
    ttl = f"Q3.1.1 Simulation of the Excitatory-Inhibitory System From {t_span[0]} To {t_span[1]}[Sec]\n"
    ttl += r'$J_{EE}$=' + f'{J_EE}, ' + r'$J_{EI}$=' + f'{J_EI}. #Of Sample Points: {NPOINTS}'
    fig.suptitle(ttl)
    ax.plot(solution.t, solution.y[0], label=r'$r_{E}$', linewidth=2)
    ax.set_xlabel('Time [Seconds]')
    ax.set_ylabel(r'$r_{E}$', rotation=0)
    ax.set_facecolor('lightgray')
    ax.grid()

    return fig, ax, solution

def find_freqs_peak(freqs, power, threshold=0.05):
    f, p = freqs[0<=freqs], np.abs(power)[0<=freqs]
    # Discard frequencies below the threshold
    f, p = f[f>threshold], p[f>threshold]
    # Find the peak frequency and its power
    return f[p.argmax()], p.max()

def q_3_1_2(sol=None, t_span=(0, 100), NPOINTS=100):
    if sol is None:
        _, _, sol = q_3_1_1(t_span=t_span, NPOINTS=NPOINTS)
    
    # Run scipy fft on solution.y[0]
    r_E_fft = fft(sol.y[0])
    # calculate the mean sampling rate
    dt = np.diff(sol.t).mean()
    # get the freqs 
    freqs = fftfreq(len(r_E_fft), d=dt)

    # Plot the power spectrum
    fig_fft, psd = plt.subplots()
    psd.plot(freqs[0<=freqs], np.abs(r_E_fft)[0<=freqs])
    psd.set_xlabel('Frequency [Hz]')
    psd.set_ylabel('Power')
    psd.set_title(
        r'Power Spectrum Of $r_{E}$' + f'\nMean Sample Spacing: {dt:.2f}[Sec]'
    )

    max_freq, max_power = find_freqs_peak(freqs, r_E_fft)
    
    psd.vlines(
        max_freq, ymin=0, ymax=max_power, 
        color='orange', linestyle='--', linewidth=2, 
        label=f'Peak Frequency at ~ {max_freq:.2f}[Hz]'
    )

    psd.grid()
    psd.legend()
    psd.set_facecolor('lightgray')
    return fig_fft, psd

def q_3_1_2_4():
    funcs = []
    y0 = np.array((2.5, 3.5))
    h_0 = np.array([36, 4])
    t_span, NPOINTS = (0, 100), 100
    for J_EE in [2,2.5,3]:
        func = []
        for J_EI in np.linspace(2, 10, 50):
            sol = simulate_excitatory_inhibitory_system(
                J_EE, J_EI, h_0, initial_conditions=y0, 
                time_span=t_span, points_num=NPOINTS
            )
            r_E_fft = fft(sol.y[0])
            freqs = fftfreq(len(r_E_fft), d=np.diff(sol.t).mean())
            max_freq, _ = find_freqs_peak(freqs, r_E_fft)
            func.append(np.array([J_EI, max_freq]))
            
        funcs.append(np.array(func))
    
    funcs = np.array(funcs)

    fig, ax = plt.subplots(figsize=(10,6))
    for i, func in enumerate(funcs):
        ax.plot(func[:,0], func[:,1], label=r'$J_{EE}$'+f'={2+(i/2)}')
    ax.set_xlabel(r'$J_{EI}$')
    ax.set_ylabel('Peak Frequency [Hz]')
    ax.grid()
    ax.legend()
    ax.set_facecolor('lightgray')

    ttl = r'Peak Frequency As A Function Of $J_{EI}$' + '\n'
    ttl += f'h_0={h_0}, y0={y0}, Times: {t_span}[Sec], #Of Sample Points: {NPOINTS}'
    fig.suptitle(ttl)

    return fig, ax

def q_3_2_1_mat_factory(J=1):
    return np.array([
        [0, J],
        [(-1 * J), 0]
    ])

def q_3_2_1_simulator(J=1, h_0=np.array([1, 1]), y0=np.array([1, 1]),
        t_span=(0, 100), NPOINTS=100, mat=None):
    if mat is None:
        M = q_3_2_1_mat_factory(J)
    else:
        M = mat
    sol = get_simulation_results(M, h_0, y0, t_span, NPOINTS)
    return sol

def q_3_2_1_fft_helper(sol):
    sol_fft = fft(sol.y[0])
    # calculate the mean sampling rate
    dt = np.diff(sol.t).mean()
    # get the freqs 
    freqs = fftfreq(len(sol_fft), d=dt)

    max_freq, _ = find_freqs_peak(freqs, sol_fft)
    return max_freq

def q_3_2_1():
    max_freqs = []
    t = np.arange(0.3, 3, 0.1)
    for J in t:
        sol = q_3_2_1_simulator(J)
        max_freq = q_3_2_1_fft_helper(sol)
        max_freqs.append(max_freq)
    
    max_freqs = np.array(max_freqs)
    fig, ax = plt.subplots()
    ax.plot(t, max_freqs, linewidth=2, label='Peak Frequency From Simulation')
    ax.plot(
        t, t/(2*np.pi), linestyle='--', color='red',
        label=r'k=$\frac{J}{2\pi}$', alpha=0.5, linewidth=2
    )
    ax.set_xlabel('J')
    ax.set_ylabel('k[Hz]', rotation=0)  
    ax.grid()
    ax.legend()
    ax.set_facecolor('lightgray')

    ttl = 'Peak Frequency As A Function Of J - Simulation VS Analytical'
    fig.suptitle(ttl)

def q_3_2_2_matrix_analyzer(
    J_EE=2, J_EI=2, y0=np.array((2.5, 3.5)),
    h_0=np.array([36, 4]), t_span=(0, 100), NPOINTS=100
):
    M = M_1(J_EE=J_EE, J_EI=J_EI)
    eigenvalues = np.linalg.eigvals(M)

    # get the imaginary part of the eigenvalues
    imaginary_eigenvalue_part = np.imag(eigenvalues)[0]
    excepted_k = imaginary_eigenvalue_part / (2 * np.pi)

    t_span, NPOINTS = (0, 100), 100

    sol = q_3_2_1_simulator(h_0=h_0, y0=y0, t_span=t_span, NPOINTS=NPOINTS, mat=M)
    max_freq = q_3_2_1_fft_helper(sol)
    
    return max_freq, excepted_k

def q_3_2_2():
    J_EIs = np.linspace(3.5, 7, 50)
    funcs = []
    for J_EE in [2,2.5,3]:
        func = []
        for J_EI in J_EIs:
            max_freq, excepted_k = q_3_2_2_matrix_analyzer(J_EE=J_EE, J_EI=J_EI)
            func.append(np.array([max_freq, excepted_k]))
        funcs.append(np.array(func))
    
    funcs = np.array(funcs)
    fig, ax = plt.subplots(figsize=(10,6))
    for i, func in enumerate(funcs):
        ax.scatter(J_EIs, func[:,0], label=r'$J_{EE}$'+f'={2+(i/2)} FFT', marker='+')
        ax.plot(J_EIs, func[:,1], label=r'$J_{EE}$'+f'={2+(i/2)} Expected')
    
    ax.set_xlabel(r'$J_{EI}$')
    ax.set_ylabel('K[Hz]', rotation=0)
    ax.grid()
    ax.legend()
    ax.set_facecolor('lightgray')

    ttl = r'Q3.2.2 Frequency Of Oscillations From The Linear Model' + '\n'
    ttl += r'Predicted VS Simulated As A Function Of $J_{EI}$'
    fig.suptitle(ttl)

    return fig, ax

def q_2():
    return [
        q_2_2_1_real_stable(),
        q_2_2_1_real_unstable(),
        q_2_2_1_complex_stable(),
        q_2_2_1_complex_unstable(),
    ]

def q_3():
    # _, _, sol_q_3_1 = q_3_1_1()
    q_3_1_2() # Includes q_3_1_1
    q_3_1_2_4()
    q_3_2_1()
    q_3_2_2()


if __name__ == "__main__":
    q_2()
    plt.show()
    q_3()
    plt.show()
    q_2_bonus()
    plt.show()
    