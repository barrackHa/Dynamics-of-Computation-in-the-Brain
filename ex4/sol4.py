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
        self.sol = None
        return

    def theta_of_i(self, i: int) -> float:
        return (2 * np.pi * i) / self.N
    
    def _init_connectivity_matrix(self) -> np.ndarray:
        thetas = self.angles
        v = (thetas < np.pi/2).astype(int) + (((3 * np.pi) / 2) < thetas).astype(int)
        self.J_ij = (1/self.N) * circulant(v)
        return self.J_ij
    
    def linear_dynamic(self, t, r):
        return -r + (self.J_ij @ r) + self.h_vec
    
    def simulate(self, t_span=(0, 1000), r0=None, dt=1):
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

        fig, ax = plt.subplots()
        if center_around_zero:
            xx = np.rad2deg(self.angles - np.pi)
        else:
            xx = np.rad2deg(self.angles)

        for t in timesteps:
            yy = self.sol.y[:,t]
            if center_around_zero:
                yy = np.roll(yy, self.N//2)
            ax.plot(xx, yy, label=f't={t}', linewidth=2)
        
        hv = np.roll(self.h_vec, self.N//2) if center_around_zero else self.h_vec
        ax.plot(xx, hv, label=r'$h_i$', linestyle='--', color='black')
        ax.set_xticks((np.arange(-180, 181, 45)))
        ax.set_xlabel('Angle on the ring [degrees]')
        ax.set_ylabel(r'$r(t,\theta)$', rotation=0, labelpad=20)
        ax.set_title('Ring model simulation')
        ax.set_facecolor('lightgray')
        ax.legend()
        # plt.show()
        return fig, ax

    

# def theta_of_i(i: int) -> float:
#     return (2 * np.pi * i) / N

#%%
if __name__ == "__main__":
    # v = np.arange(1,5)
    # print(circulant(v))
    N = 1001
    # print([np.rad2deg(theta_of_i(i)) for i in range(N)])
    # thets= np.array([(theta_of_i(i)) for i in range(N)])
    # print(np.abs(thets - np.pi) > np.pi/2)
    # v = (thets < np.pi/2).astype(int) + (((3 * np.pi) / 2)
    ring = RingModel(N=N)
    dur = ring.simulate()
    print(f"Simulation took {dur} seconds")
    t = [1, 25, 50, 200, 600]
    fig, ax = ring.plot_sim(t)
    
    # print(sol.t.shape, sol.y.shape)
    # xx = np.rad2deg(ring.angles - np.pi)
    # yy = np.roll(sol.y[:,t], N//2)
    # plt.plot(np.rad2deg(ring.angles), sol.y[:,-300])
    # for y in yy.T:
    #     plt.plot(xx, y)
    # plt.xticks((np.arange(-180, 181, 45)))
    plt.show()
    # print(np.rad2deg(ring.angles))
    # print(np.roll(np.rad2deg(ring.angles), shift=N//2))
    # # change ring.angles from radians between 0-2pi to between -pi to pi so that positive valuse bigger than pi will be negative
    # angles = ring.angles - np.pi
    # print(np.rad2deg(angles))

    
    # ring
    
    
# %%
