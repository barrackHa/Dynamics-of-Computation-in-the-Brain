"""
Solution for exercise 3.
Dynamcs Of Computation In The Brain - 76908
By: Barak H.
June 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks

def f(v):
    return v - ((v**3) / 3)

def fhn(t, y, I_ext=0.8, epsilon=0.08, a=0.7, b=0.8, dt=0.1, t1=None):
    v, w = y
    if (t1 is not None):
        perturbation_time = (t1 * dt)
        if (perturbation_time < t) and (t <= (perturbation_time + (2*dt))):
            v = v + np.abs(v * 0.1)
            w = w + np.abs(w * 0.1)
            print(f't={t}, v={v}, w={w}')

    dv_dt = f(v) - w + I_ext
    dw_dt = epsilon * (v + a - (b * w))
    return [dv_dt, dw_dt]

def get_sim_results(
    y0=[0.1, 0.1], t_span=(0, 200), dt=0.1, I_ext=0.8, 
    epsilon=0.08, a=0.7, b=0.8, t1=None
):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        fhn, t_span, y0, t_eval=t_eval, 
        args=(I_ext, epsilon, a, b, dt, t1), method='LSODA'
    )
    return sol

def fig_ax_formater(fig, ax):
    ax.set_xlabel('v')
    ax.set_ylabel('w', rotation=0)
    ax.set_title('I_ext=0.8, ' + r'$\epsilon$=' + '0.08, a=0.7, b=0.8')
    ax.legend()
    ax.grid()
    ax.set_facecolor('lightgray')
    fig.suptitle(f'FitzHugh-Nagumo Phase Plane')
    return fig, ax

def q_2_1_1():
    """How initial conditions affect the phase plane trajectory."""
    fig, ax = plt.subplots()
    for y0 in [[0.1, 0.1], [2, 3], [-1, -3]]:
        fhn_sol = get_sim_results(y0=y0)
        ax.plot(fhn_sol.y[0], fhn_sol.y[1], label=f'y0={y0}')
        sc = ax.scatter(fhn_sol.y[0], fhn_sol.y[1], c=fhn_sol.t, cmap='viridis', alpha=0.7)
    
    fig.colorbar(sc, ax=ax, label='Time scale')
    fig, ax = fig_ax_formater(fig, ax) 
    return fig, ax

def q_2_1_2():
    """How different external currents affect the phase plane trajectory."""
    fig, ax = plt.subplots()
    y0 = [0.1, 0.1]
    for i_ext in [0, 0.6, 1, 1.5, 2, 4, 10]:
        fhn_sol = get_sim_results(y0=y0, I_ext=i_ext)
        ax.plot(fhn_sol.y[0], fhn_sol.y[1], label=r'$I_{ext}$='+f'{i_ext}')
        sc = ax.scatter(fhn_sol.y[0], fhn_sol.y[1], c=fhn_sol.t, cmap='viridis', alpha=0.7)
    
    fig.colorbar(sc, ax=ax, label='Time scale')
    fig, ax = fig_ax_formater(fig, ax)
    ax.set_title(f'Initial condition: {y0}, ' + r'$\epsilon$=' + '0.08, a=0.7, b=0.8')
    return fig, ax

def q_2_1_3():
    """How different values of b affect the phase plane trajectory."""
    fig, ax = plt.subplots()
    y0 = [0.1, 0.1]
    for b in np.linspace(0.8, 2, 5):
        fhn_sol = get_sim_results(y0=y0, b=b)
        ax.plot(fhn_sol.y[0], fhn_sol.y[1], label=f'b={b}')
        sc = ax.scatter(fhn_sol.y[0], fhn_sol.y[1], c=fhn_sol.t, cmap='viridis', alpha=0.7)
    
    fig.colorbar(sc, ax=ax, label='Time scale')
    fig, ax = fig_ax_formater(fig, ax)
    ax.set_title(f'Initial condition: {y0}, ' + r'$\epsilon$=' + f'0.08, a=0.7')
    return fig, ax

def q_2_1():
    q_2_1_1()
    q_2_1_2()
    q_2_1_3()

def find_freqs_peak(freqs, power, threshold=0.001):
    """Find the peak frequency and its power."""
    f, p = freqs[0<=freqs], np.abs(power)[0<=freqs]
    # Discard frequencies below the threshold
    f, p = f[f>threshold], p[f>threshold]
    # Find the peak frequency and its power
    return f[p.argmax()], p.max()

def get_fft_results(t, y):
    """FFT analysis of the given signal y(t)."""
    power = fft(y)
    # calculate the mean sampling rate
    dt = np.diff(t).mean()
    # get the freqs 
    freqs = fftfreq(len(power), d=dt)
    return freqs, power

def get_period_with_fft(t, y):
    """Find the period of the signal using the peak frequency."""
    freqs, power = get_fft_results(t, y)
    peak_freq, _ = find_freqs_peak(freqs, power)
    return 1/peak_freq

def get_period_with_peaks(t, y):
    """Find the period of the signal using the mean distance of peaks."""
    # use scipy to find the peaks
    peaks = find_peaks(y)
    dt = np.diff(t).mean()

    # For debugging purposes:
    # plt.plot(t, y)
    # plt.scatter(t[peaks[0]], y[peaks[0]], c='r')
    
    return (np.diff(peaks[0]) * dt).mean()


def q_2_2_1(y0):
    fig, ax = plt.subplots(figsize=(8, 6))
    fhn_sol = get_sim_results(y0=y0)
    ax.plot(fhn_sol.t, fhn_sol.y[0], label='v(t)', linewidth=2)
    fig, ax = fig_ax_formater(fig, ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('v', rotation=0)
    fig.suptitle(f'Q.2.2.1 FitzHugh-Nagumo V vs Time')
    ttl = f'Initial condition: {y0}, ' + r'$\epsilon$=' + f'0.08, a=0.7, b=0.8, ' + r'$I_{ext}$=0.8'
    
    peak_period_by_fft = get_period_with_fft(fhn_sol.t, fhn_sol.y[0])
    print(peak_period_by_fft)

    mean_period_by_peaks = get_period_with_peaks(fhn_sol.t, fhn_sol.y[0])
    print(mean_period_by_peaks)

    ttl += f'\nPeriod by FFT: {peak_period_by_fft:.2f} & Period by peaks: {mean_period_by_peaks:.2f}'
    ax.set_title(ttl)
    return fig, ax  

def q_2_2_2(y0, delta_v=0.25, delta_w=0.25):
    fig, ax = plt.subplots()

    fhn_sol = get_sim_results(y0=y0)
    ax.plot(fhn_sol.t, fhn_sol.y[0], label=f'Starting at {y0}', linewidth=2)
    
    pert_y0 = [(y0[0] * (1 + delta_v)), (y0[1] * (1 + delta_w))]
    pert_sol = get_sim_results(y0=pert_y0)
    ax.plot(pert_sol.t, pert_sol.y[0], label=f'Starting at {pert_y0}', linewidth=2)

    fig, ax = fig_ax_formater(fig, ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('v', rotation=0)
    fig.suptitle(f'Q.2.2.2 FitzHugh-Nagumo V vs Time')
    ax.set_title(
        f'Initial condition: {y0}, ' + r'$\epsilon$=' + f'0.08, a=0.7, b=0.8, ' + r'$I_{ext}$=0.8'
    )

    return fig, ax    

def q_2_2():
    y0 = [1, 0.1]
    q_2_2_1(y0)
    q_2_2_2(y0)


def q_2_3_1():
    fig, ax = plt.subplots()
    y0 = [0.8,0.1]
    for t1 in [None, 400]:
        fhn_sol = get_sim_results(t1=t1, y0=y0)
        ax.plot(fhn_sol.t, fhn_sol.y[0], label=f't1={t1}')
    # ax.axvline(40, color='r', linestyle='--', label=r'$t_{1}$')
    fig, ax = fig_ax_formater(fig, ax)
    ax.set_xlabel('t')  
    ax.set_ylabel('v', rotation=0)
    fig.suptitle(f'Q.2.3.1 FitzHugh-Nagumo V vs Time')
    ax.set_title(
        f'Initial condition: {y0}, ' + r'$\epsilon$=' + f'0.08, a=0.7, b=0.8' + r'$I_{ext}$=0.8'
    )
    return fig, ax


def q_2_3_2():
    fig, ax = plt.subplots()
    y0 = [0.8,0.1]
    for t1 in [300, 400, 500, 600, 680]:
        fhn_sol = get_sim_results(t1=t1, y0=y0)
        ax.plot(fhn_sol.t, fhn_sol.y[0], label=f't1={t1}')
        # ax.vlines(t1*0.1, ymin=fhn_sol.y[0].min(), ymax=fhn_sol.y[0, t1], linestyle='--', label=r'$t_{1}$')
    fig, ax = fig_ax_formater(fig, ax)
    ax.set_xlabel('t')  
    ax.set_ylabel('v', rotation=0)
    fig.suptitle(f'Q.2.3.2 FitzHugh-Nagumo V vs Time')
    ax.set_title(
        f'Initial condition: {y0}, ' + r'$\epsilon$=' + f'0.08, a=0.7, b=0.8, ' + r'$I_{ext}$=0.8'
    )
    return fig, ax

def q_2_3():
    q_2_3_1()
    q_2_3_2()

def coupled_fhn(t, y, I_ext=0.8, epsilon=0.08, a=0.7, b=0.8, dt=0.1, gamma=0.4, t1=None):
    v1, w1, v2, w2 = y
    if (t1 is not None):
        perturbation_time = (t1 * dt)
        if (perturbation_time < t) and (t <= (perturbation_time + (6 * dt))):
            v2 = v1 + np.abs(v1 * 0.1)
            w2 = w2 + np.abs(w1 * 0.1)
            print(f't={t}, v1={v1}, w1={w1}')

    dv1_dt = f(v1) - w1 + I_ext + (gamma * (v1 - v2))
    dv2_dt = f(v2) - w2 + I_ext + (gamma * (v2 - v1))

    dw1_dt = epsilon * (v1 + a - (b * w1))
    dw2_dt = epsilon * (v2 + a - (b * w2))
    return [dv1_dt, dw1_dt, dv2_dt, dw2_dt]

def get_coupled_fhn_sim_results(
    y0=[0.1, 0.1, 3, 0.2], t_span=(0, 200), dt=0.1, I_ext=0.8, 
    epsilon=0.08, a=0.7, b=0.8, gamma=0.2, t1=None
):
    t_eval = np.arange(t_span[0], t_span[1], dt)
    sol = solve_ivp(
        coupled_fhn, t_span, y0, t_eval=t_eval, 
        args=(I_ext, epsilon, a, b, dt, gamma, t1), method='LSODA'
    )
    return sol

def coupled_phase_formater(fig, ax):
    ax.set_xlabel('v')
    ax.set_ylabel('w', rotation=0)
    ax_ttl = r'$I_{ext}$=0.8, $\epsilon$=' + '0.08, a=0.7, b=0.8, ' + r'$\gamma$=0.2'
    ax_ttl += '\nv1=0.1, w1=0.2 & v2=0.4, w2=0.5'
    ax.set_title(ax_ttl)
    ax.legend()
    ax.grid()
    ax.set_facecolor('lightgray')
    fig.suptitle(f'Q.2.4: Coupled oscillators - FitzHugh-Nagumo Phase Plane')
    return fig, ax

def v_to_t_formater(fig, ax):
    fig, ax = coupled_phase_formater(fig, ax)
    ax.set_xlabel('Time')
    ax.set_ylabel('v')
    return fig, ax

def q_2_4_1():
    phas_fig, phas_ax = plt.subplots()
    y0 = [0.1, 0.2, 0.4, 0.5]
    sol = get_coupled_fhn_sim_results(y0=y0)
    phas_ax.plot(sol.y[0], sol.y[1], label='Neuron #1')    
    phas_ax.plot(sol.y[2], sol.y[3], label='Neuron #2')
    sc = phas_ax.scatter(sol.y[0], sol.y[1], c=sol.t, cmap='viridis', alpha=0.7)
    sc = phas_ax.scatter(sol.y[2], sol.y[3], c=sol.t, cmap='viridis', alpha=0.7)
    phas_fig.colorbar(sc, ax=phas_ax, label='Time scale')

    phas_fig, phas_ax = coupled_phase_formater(phas_fig, phas_ax)

    period  = get_period_with_fft(sol.t, sol.y[0])
    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='v1(t)', linewidth=2)
    print(period)
    ax.plot(sol.t, sol.y[2], label='v2(t)', linewidth=2)

    fig, ax = v_to_t_formater(fig,ax)

    return phas_fig, phas_ax, fig, ax

def q_2_4_2(perturb=False):
    phas_fig, phas_ax = plt.subplots()
    y0 = [0.1, 0.2, 0.1, 0.2]

    if perturb:
        sol = get_coupled_fhn_sim_results(t1=300, y0=y0)
    else:
        sol = get_coupled_fhn_sim_results(y0=y0)
    
    n1_period = get_period_with_peaks(sol.t[1000:], sol.y[0,1000:])
    n1_fft = get_period_with_fft(sol.t[1000:], sol.y[0,1000:])
    n_1_cycle = np.mean([n1_period, n1_fft])
    n2_period = get_period_with_peaks(sol.t[1000:], sol.y[2,1000:])
    n2_fft = get_period_with_fft(sol.t[1000:], sol.y[2,1000:])
    n_2_cycle = np.mean([n2_period, n2_fft])

    print(n1_period, n1_fft)
    print(n2_period, n2_fft)

    phas_ax.plot(sol.y[0], sol.y[1], label='Neuron #1')    
    phas_ax.plot(sol.y[2], sol.y[3], label='Neuron #2')

    sc = phas_ax.scatter(sol.y[2], sol.y[3], c=sol.t, cmap='viridis', alpha=0.7)
    sc = phas_ax.scatter(sol.y[0], sol.y[1], c=sol.t, cmap='viridis', alpha=0.7)
    phas_fig.colorbar(sc, ax=phas_ax, label='Time scale')

    coupled_phase_formater(phas_fig, phas_ax)
    ax_ttl = r'$I_{ext}$=0.8, $\epsilon$=' + '0.08, a=0.7, b=0.8, ' + r'$\gamma$=0.2'
    ax_ttl += '\nv1=0.1, w1=0.2 & v2=0.1, w2=0.2'
    phas_ax.set_title(ax_ttl)

    fig, ax = plt.subplots()
    ax.plot(sol.t, sol.y[0], label='v1(t)', linewidth=2)
    ax.plot(sol.t, sol.y[2], label='v2(t)', linewidth=2)

    fig, ax = v_to_t_formater(fig,ax)
    fig.suptitle('Q.2.4 V vs Time - Neurons 1 & 2')
    ax.set_title(f'Neuron 1: Period ~ {n_1_cycle:.2f} & Neuron 2: Period ~ {n_2_cycle:.2f}')

    return phas_fig, phas_ax, fig, ax

def q_2_4():
    q_2_4_1()
    q_2_4_2()
    q_2_4_2(perturb=True)

if __name__ == "__main__":
    q_2_1()
    plt.show()
    q_2_2()
    plt.show()
    q_2_3()
    plt.show()
    q_2_4()
    plt.show()