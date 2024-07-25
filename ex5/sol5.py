"""
Solution for exercise 4.
Dynamcs Of Computation In The Brain - 76908
By: Barak H.
July 2024
"""
#%%
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

rng = np.random.default_rng(22040723)
# set globaly the font size for the plots
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6),
    'axes.labelsize': 12,
    # 'axes.facecolor': 'lightgrey',
    'axes.grid': True,
    'lines.linewidth': 3
})
# set globaly the figure size for the plots
# plt.rcParams.update({'figure.figsize': (10, 6)})
# # set globaly the axes label size for the plots
# plt.rcParams.update({'axes.labelsize': 12})
# # set globaly the axes face color for the plots
# plt.rcParams.update({'axes.facecolor': 'lightgrey'})
# # Set grid globally
# plt.rcParams.update({'axes.grid': True})
# # Set line width globally
# plt.rcParams.update({'lines.linewidth': 3})

class HopfieldNetwork:
    def __init__(self, N=1000, f=0.5, theta=0, tau=5, beta=20):
        self.N = N
        self.f = f
        self.theta = theta
        self.tau = tau
        self.dt = 0.9 * (self.tau / 20)
        self.beta = beta
        self.P = None
        self.J = None
        self.patterns = None
        
    def generate_random_patterns(self, num_patterns):
        """Generate random patterns for the network"""
        self.P = num_patterns
        self.patterns = rng.choice(
            [0, 1], size=(num_patterns, self.N), p=[1-self.f, self.f], 
        )
        # self.patterns shape is (#num_patterns, #self.N entries in each pattern)
        return self.patterns
    
    def train(self, patterns=None):
        """Calculate the weights matrix J from the patterns"""
        if patterns is not None:
            self.patterns = patterns
        elif self.patterns is None:
            raise ValueError("No patterns to train on")
        
        self.J = np.zeros((self.N, self.N))
        for pattern in self.patterns:
            self.J += np.outer((pattern - self.f), (pattern - self.f))
        self.J /= (self.N * self.f * (1 - self.f))
        # Set the diagonal elements to zero (no self-connections)
        self.J *= np.ones_like(self.J) - np.eye(self.N)
        return self.J
    
    def nonlinearity(self, x, theta=None, beta=None):
        """g(x) - the logistic function is the nonlinearity function of the neuron"""
        if theta is None:
            theta = self.theta
        if beta is None:
            beta = self.beta
        return 1 / (1 + np.exp(beta * (theta - x)))
    
    def dynamics(self, t, r):
        """The dynamics of the network"""
        return -r + self.nonlinearity(self.J @ r)
    
    def simulate(self, r0=None, t_span=(0, 10)):
        """Simulate the dynamics of the network"""
        self.sol = solve_ivp(
            self.dynamics, t_span, r0, 
            t_eval=np.arange(*t_span, self.dt), method='LSODA'
        )
        return self.sol
    
    def get_error(self, vec1, vec2, threshold=0.1):
        """Calculate the number of different neurons between two neural state vectors"""
        return ((vec1 - vec2) > threshold).sum() / self.N

    def get_probable_error(self, P_smpl=20, threshold=0.1, T_frac=0.25):
        """Calculate the fraction of times the simulation gets the wrong neurons"""
        # get #P_smpl uniqe samples out of arange(self.N)
        smpls = rng.choice(self.P, P_smpl, replace=True)
        # select random sample of memory patterns
        smpl_patterns = self.patterns[smpls, :]
        errs_over_samples = np.zeros(P_smpl)
        # for each of these memory patterns start the network from 
        # the initial condition of that memory pattern
        for i, smpl in enumerate(smpl_patterns):
            # run the dynamics for short time T = 0.25τ  
            sol = self.simulate(r0=smpl, t_span=(0, T_frac*self.tau))
            # check the fraction of neurons that have errors 
            errs_over_samples[i] = self.get_error(sol.y[:,-1], smpl, threshold)
        return np.sum(errs_over_samples) / P_smpl
    
    def overlap_of_mem_patterns(self, r, mu=None):
        """
        Calculate the overlap between vector r to the mu'th mem 
        pattern (if mu is None, to all patterns).
        """
        c = (1 / (self.N * self.f * (1 - self.f)))
         # (P X N) @ (N X time_steps) = (P X time_steps)
        ovrlp_with_patterns = c * ((self.patterns - self.f) @ r)
        if mu is not None:
            ovrlp_with_patterns = ovrlp_with_patterns[mu]
        return ovrlp_with_patterns
    
def sim_run_helper(P, N, T_frac=0.25):
    hn = HopfieldNetwork(N)
    _ = hn.generate_random_patterns(P)
    _ = hn.train()
    # smp_size = np.min([20, P//2])
    smp_size = 20
    probable_error = hn.get_probable_error(P_smpl=smp_size, T_frac=T_frac)
    return hn, probable_error

def get_probable_errors(Ps, T_frac, N=1000):
    trained_networks = Parallel(n_jobs=-1)(
        delayed(sim_run_helper)(P, N, T_frac) for P in Ps
    )
    probable_errors = [res[-1] for res in trained_networks]
    return probable_errors

def q_1_1(N=1000):
    Ps = [10, 20, 30, 80, 90, 100, 200, 300, 400, 500]
    T_frac=0.25
    probable_errors = get_probable_errors(Ps, T_frac, N)

    mc = 'orange'
    fig, ax = plt.subplots()
    ax.plot(Ps, probable_errors, 'o-', markerfacecolor=mc, markersize=7, label=r'T=0.25$\tau$')
    ax.set_xlabel("Number of patterns")
    ax.set_ylabel("Probable error")
    # fig.suptitle("Probable error as a function of the number of patterns")
    # ax.set_title(f"{N} Nuerons. Ran for {0.25}" + r"$\tau$")
    ax.legend()
    return fig, ax

def q_1_2_sim_helper(beta, N=1000, P=50):
    hn = HopfieldNetwork(N=N, beta=beta)
    _ = hn.generate_random_patterns(P)
    _ = hn.train()
    sol = hn.simulate(r0=hn.patterns[0], t_span=(0, 50*hn.tau))
    patterns_ovlp = hn.overlap_of_mem_patterns(sol.y)
    return patterns_ovlp, sol, hn

def q_1_2_1(N=1000, P=50):
    betas = [1, 4, 6, 8, 10, 15]
    mem_ovrlps_by_beta = Parallel(n_jobs=-1)(
        delayed(q_1_2_sim_helper)(beta=b, N=N, P=P) for b in betas
    )
    # tot_ovlp, sol, _ = q_1_2_sim_helper(beta=1, N=N, P=P)
    # tot_ovlp, sol, _ = mem_ovrlps_by_beta[0]

    fig, axes = plt.subplots(3,2, figsize=(15, 8), sharex=True, sharey=True)
    for i, (beta, (tot_ovlp, sol, _)) in enumerate(zip(betas, mem_ovrlps_by_beta)):
        ax = axes[i%3, i//3]
        ax.plot(sol.t, tot_ovlp[1:].T, color='skyblue', alpha=0.5, linewidth=1)
        ax.plot(sol.t, tot_ovlp[0], label="Overlap with 1st pattern")
        # ax.set_xlabel("Time")
        ax.set_xticks([])
        # ax.set_ylabel("Overlap\nscore", rotation=0, labelpad=20)
        ax.set_title(r"$\beta$ = " + f"{beta}")
        
    print(sol.t.shape, sol.t[0], sol.t[-1])
    # axes[2,0].set_xticks(np.linspace(0, np.round(sol.t[-1]), 5))
    # axes[2,1].set_xticks(np.arange(0, sol.t.size, 10))
    fig.suptitle("Overlap score of the network with the memory patterns")
    return fig, axes

def q_1_2(N=1000, P=50):
    # fig, axs = q_1_2_1()
    betas = [1, 2, 4, 6, 8, 10, 15, 20, 25]
    res = np.zeros((10, len(betas)))
    print(res.shape)
    # mem_ovrlps_by_beta = Parallel(n_jobs=-1)(
    #     delayed(q_1_2_sim_helper)(beta=b, N=N, P=P) for b in betas
    # )
    # for m in mem_ovrlps_by_beta:
    #     print(m[0][0,-1])
    def tmp(i):
        mem_ovrlps_by_beta = Parallel(n_jobs=-1)(
            delayed(q_1_2_sim_helper)(beta=b, N=N, P=P) for b in betas
        )
        res[i][:] = np.array([m[0][0,-1] for m in mem_ovrlps_by_beta])

    # for i in range(2):
    #     mem_ovrlps_by_beta = Parallel(n_jobs=-1)(
    #         delayed(q_1_2_sim_helper)(beta=b, N=N, P=P) for b in betas
    #     )
    #     res[i][:] = np.array([m[0][0,-1] for m in mem_ovrlps_by_beta])
    
    fig, ax = plt.subplots()
    ax.plot(betas, res.mean(axis=0), 'o-', label="Mean")
    ax.fill_between(betas, res.min(axis=0), res.max(axis=0), alpha=0.3, label="Min-Max")

    return

#%%
if __name__ == "__main__":
    P, N = 50, 1000
    # q_1_1(N)
    # plt.show()
    q_1_2(N, P)
    plt.show()
    exit()
    hn = HopfieldNetwork(N)
    p = hn.generate_random_patterns(P)
    print(hn.patterns[0], hn.patterns.shape)
    J = hn.train()
    # print(hn.J, hn.J.shape)

    sol = hn.simulate(r0=hn.patterns[0], t_span=(0, 0.25*hn.tau))
    # print(sol.y, sol.y.shape)
    print(((sol.y[:,-1] - hn.patterns[0]) > 0.1).sum())
    get_probable_error = hn.get_probable_error()
    print(get_probable_error)


