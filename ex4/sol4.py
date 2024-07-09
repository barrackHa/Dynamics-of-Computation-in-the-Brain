"""
Solution for exercise 4.
Dynamcs Of Computation In The Brain - 76908
By: Barak H.
July 2024
"""
#%%
import numpy as np
from scipy.linalg import circulant
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

class RingModel:
    def __init__(self, N=1001, J=1, h_const=2) -> None:
        # Number of neurons in the ring
        self.N = N
        # Coupling strength of connected neurons
        self.J = J
        # h_i = h_const if (|pi - theta_i| < pi/2), 0 otherwise
        self.h_const = h_const
        # Prefered angles of neurons
        self.angles = np.array([(self.theta_of_i(i)) for i in range(self.N)])
        # vector of h_i values
        self.h_vec = (np.abs(np.pi - self.angles) > (np.pi/2)).astype(int) * 2
        # Connectivity matrix
        self.J_ij = self._init_connectivity_matrix()
        # Numerical solution of the simulation
        self.sol = None
        return

    def theta_of_i(self, i: int) -> float:
        return (2 * np.pi * i) / self.N
    
    def _init_connectivity_matrix(self) -> np.ndarray:
        thetas = self.angles
        v = (thetas < np.pi/2).astype(int) + (((3 * np.pi) / 2) < thetas).astype(int)
        self.J_ij = (1/self.N) * self.J * circulant(v)
        return self.J_ij
    
    def linear_dynamic(self, t, r):
        return -r + (self.J_ij @ r) + self.h_vec
    
    def simulate(self, t_span=(0, 1000), r0=None, dt=0.05):
        if r0 is None:
            r0 = np.ones(self.N) * 0.1
        start = time.time()
        sol = solve_ivp(
            self.linear_dynamic, t_span, r0, t_eval=np.arange(*t_span, dt), 
            method='LSODA'
        )
        end = time.time()
        self.sol = sol
        return end-start
    
    def plot_sim(self, timesteps, center_around_zero=True):
        if self.sol is None:
            raise ValueError("No simulation was done")
        
        try:
            timesteps[0]
        except TypeError:
            timesteps = np.array([timesteps])

        if center_around_zero:
            xx = np.rad2deg(self.angles - np.pi)
        else:
            xx = np.rad2deg(self.angles)
        
        fig, ax = plt.subplots()
        for t in timesteps:
            yy = self.sol.y[:,t]
            if center_around_zero:
                yy = np.roll(yy, self.N//2)
            ax.plot(xx, yy, label=f't={t}', linewidth=3)
        
        hv = np.roll(self.h_vec, self.N//2) if center_around_zero else self.h_vec
        ax.plot(xx, hv, label=r'$h(\theta)$', linestyle='--', color='black', linewidth=3)
        x_tick_names = [r'-$\pi$', r'-$\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$']
        ax.set_xticks((np.arange(-180, 181, 90)), x_tick_names)
        ax.set_xlabel('Angle on the ring [Rad]')
        ax.set_ylabel(r'$r(\theta,t)$', rotation=0, labelpad=20)
        fig.suptitle('Ring model simulation')
        mean_dt = np.round(np.diff(self.sol.t).mean(), decimals=2)
        t0, t_end = self.sol.t[0], np.rint(self.sol.t[-1])
        ax.set_title(f'N={self.N}, J={self.J}, dt={mean_dt}, Time=[{t0:.2f}, {t_end:.2f}] [sec]')
        ax.set_facecolor('lightgray')
        ax.legend(fontsize=14)
        return fig, ax
#%%
def q_2_1(N=1001):
    ring = RingModel(N=N)
    dur = ring.simulate()
    print(f"Simulation took {np.round(dur, decimals=2)} seconds")
    times = [1, 25, 50, 200, 600]
    fig, ax = ring.plot_sim(times)
    return fig, ax, dur

def sim(N, J):
    ring = RingModel(N=N, J=J)
    _ = ring.simulate()
    return ring

def run_q_2_2_sims(N=1001, Js=[1, 1.5, 1.9, 2.1]):
    start = time.time()
    results = Parallel(n_jobs=-1)(
        delayed(sim)(N, J) for J in Js
    )
    end = time.time()
    print(f"Simulation took {np.round(end-start, decimals=2)} seconds")
    return results

def q_2_2(N=1001, zoomin=True):
    # Run the simulations
    res = run_q_2_2_sims(N)

    # Calculate the mean activity across the ring
    mean_r_across_ring = np.array(
        [np.mean(r.sol.y, axis=0) for r in res]
    )

    # Plot the mean activity across the ring vs time
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, J in enumerate([1, 1.5, 1.9, 2.1]):
        t = res[i].sol.t
        ax.plot(t, mean_r_across_ring[i], label=f'J={J}', linewidth=3)
    ax.set_title('Mean activity across the ring', fontsize=16)
    ax.set_xlabel('Time [sec]', fontsize=10)
    ax.set_ylabel('Mean r(t)', rotation=0, labelpad=20, fontsize=10)
    if zoomin:
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 25])
    ax.set_facecolor('lightgray')
    ax.grid()
    ax.legend(fontsize=14)

    # Plot the simulation for J=2.1
    divergence_fig, divergence_ax = res[-1].plot_sim(timesteps=[1, 25, 50, 200, 600])

    return fig, ax, divergence_fig, divergence_ax

#%%
def q_3(Ns=[3, 5, 49, 257]):
    for N in Ns:
        fig, _, divergence_fig, _ = q_2_2(N, zoomin=False)
        fig.suptitle(f'N={N}', fontsize=16)
        divergence_fig.clf()
        plt.close(divergence_fig)
    return 
#%%
if __name__ == "__main__":
    
    N = 1001
    fig, ax, dur = q_2_1(N)
    plt.show()
    fig, ax, divergence_fig, divergence_ax = q_2_2(N, zoomin=False)
    plt.show()
    q_3()
    plt.show()
