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
from scipy.special import erf

rng = np.random.default_rng(22040723)
# set globaly the font size for the plots
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6),
    'axes.labelsize': 12,
    'axes.grid': True,
    'lines.linewidth': 3
})

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
        return (-r + self.nonlinearity(self.J @ r)) / (self.tau)
    
    def simulate(self, r0=None, t_span=(0, 10)):
        """Simulate the dynamics of the network"""
        self.sol = solve_ivp(
            self.dynamics, t_span, r0, 
            t_eval=np.arange(*t_span, self.dt), method='LSODA'
        )
        return self.sol
    
    def get_error(self, vec1, vec2, threshold=0.1):
        """Calculate the number of different neurons between two neural state vectors"""
        return (np.abs(vec1 - vec2) > threshold).sum() / self.N

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
            # run the dynamics for  time T = (T_frac * τ)  
            sol = self.simulate(r0=smpl, t_span=(0, T_frac*self.tau))
            # check the fraction of neurons that have errors 
            errs_over_samples[i] = self.get_error(sol.y[:,-1], smpl, threshold)
        
        return errs_over_samples.mean(), errs_over_samples.std()
    
    def get_analytical_error_prob(self, P=None):
        if P is None:
            P = self.P
        mu = 1 - self.f - self.theta  # mean of the normal distribution
        sigma = np.sqrt((P * self.f) / N)  # standard deviation
        prob_error = 0.5 * (1 + erf((-mu) / (sigma * np.sqrt(2))))
        return prob_error
    
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
    smp_size = 20
    probable_error_mean, probable_error_std = hn.get_probable_error(
        P_smpl=smp_size, T_frac=T_frac
    )
    return hn, probable_error_mean, probable_error_std

def get_probable_errors(Ps, T_frac, N=1000):
    trained_networks = Parallel(n_jobs=-1)(
        delayed(sim_run_helper)(P, N, T_frac) for P in Ps
    )
    probable_errors = [res[1] for res in trained_networks]
    return probable_errors, trained_networks

def q_1_1(N=1000):
    Ps = np.array([10, 20, 30, 80, 90, 100, 200, 300, 400, 500])
    T_frac = 0.25
    mc = 'orange'
    fig, ax = plt.subplots()

    # 1.1.1
    probable_errors, trained_networks = get_probable_errors(Ps, T_frac, N)
    stds = np.array([res[-1] for res in trained_networks])

    # 1.1.2
    examplar_hn = trained_networks[0][0]
    error_exp_cdf = examplar_hn.get_analytical_error_prob(Ps)
    
    # plot 1.1.1
    ax.plot(Ps, probable_errors, 'o-', markerfacecolor=mc, markersize=7, label=r'T=0.25$\tau$')
    ax.fill_between(Ps, probable_errors - stds, probable_errors + stds, alpha=0.3)

    # plot 1.1.2
    ax.plot(Ps, error_exp_cdf, '-', label=r'Analytical', color='black')

    # 1.1.3
    for frac in [2, 18]:
        T_frac = frac
        probable_errors, trained_networks = get_probable_errors(Ps, T_frac, N)
        stds = np.array([res[-1] for res in trained_networks])
        ax.plot(Ps, probable_errors, 'o-', markerfacecolor=mc, markersize=7, label=f'T={frac}' + r'$\tau$')
        ax.fill_between(Ps, probable_errors - stds, probable_errors + stds, alpha=0.3)

    ax.set_xlabel("Number of patterns")
    ax.set_ylabel("Probable error")
    fig.suptitle("Q.1.1: Probability of an error as a function of the number of patterns")
    f, theta, beta, tau = examplar_hn.f, examplar_hn.theta, examplar_hn.beta, examplar_hn.tau
    ax.set_title(
        f"{N} Nuerons, f={f}, " + r"$\tau$=" + f'{tau}, ' \
        + r"$\theta$=" + f'{theta}, ' + r"$\beta$= " + f'{beta}'
    )
    ax.legend()
    return fig, ax

def q_1_2_sim_helper(beta, N=1000, P=50):
    hn = HopfieldNetwork(N=N, beta=beta)
    _ = hn.generate_random_patterns(P)
    _ = hn.train()
    sol = hn.simulate(r0=hn.patterns[0], t_span=(0, 20*hn.tau))
    patterns_ovlp = hn.overlap_of_mem_patterns(sol.y)
    return patterns_ovlp, sol, hn

def q_1_2_1(N=1000, P=50):
    betas = [1, 4, 6, 8, 10, 15]
    mem_ovrlps_by_beta = Parallel(n_jobs=-1)(
        delayed(q_1_2_sim_helper)(beta=b, N=N, P=P) for b in betas
    )

    fig, axes = plt.subplots(3,2, figsize=(15, 8), sharex=True, sharey=False)
    for i, (beta, (tot_ovlp, sol, _)) in enumerate(zip(betas, mem_ovrlps_by_beta)):
        ax = axes[i%3, i//3]
        ax.plot(sol.t, tot_ovlp[1:].T, color='skyblue', alpha=0.5, linewidth=1)
        ax.plot(sol.t, tot_ovlp[0], label="Overlap with 1st pattern")
        ax.set_xticks([])
        ax.set_title(r"$\beta$ = " + f"{beta}")
        
    print(sol.t.shape, sol.t[0], sol.t[-1])
    t = np.linspace(0, np.round(sol.t[-1]), 5, dtype=int)
    axes[2,0].set_xticks(t, t)
    axes[2,1].set_xticks(t, t)
    axes[2,0].set_xlabel('Time'), axes[2,1].set_xlabel('Time')
    axes[1,0].set_ylabel(f'Overlap\nscore', rotation=0, labelpad=25)
    axes[2,1].legend()
    fig.suptitle("Q1.2.1: Overlap score of the network with the memory patterns")
    return fig, axes

def  q_1_2_2_n_3(N):
    betas = [1, 2, 4, 6, 8, 10, 15, 20, 25]
    inters = 10
    fig, ax = plt.subplots()
    for P in [50, 65]:
        res = np.zeros((inters, len(betas)))
        for i in range(inters):
            mem_ovrlps_by_beta = Parallel(n_jobs=-1)(
                delayed(q_1_2_sim_helper)(beta=b, N=N, P=P) for b in betas
            )
            res[i][:] = np.array([m[0][0,-1] for m in mem_ovrlps_by_beta])
        mean_values = res.T.mean(axis=1)
        stds = res.T.std(axis=1)
        ax.plot(betas, mean_values, 'o-', label=f"Mean overlap P={P}", markerfacecolor='orange', markersize=7)
        ax.fill_between(betas, (mean_values - stds), (mean_values + stds), alpha=0.3)
    
    ax.set_xlabel(r"$\beta$'s")
    ax.set_ylabel("Mean overlap with 1st pattern")
    ax.set_title("Q1.2.2+3: Mean overlap with the first memory pattern as a function of β")
    ax.legend()

    return fig, ax

def q_1_2(N=1000, P=50):
    fig, ax = q_1_2_1(N, P)
    fig2, ax2 = q_1_2_2_n_3(N)
    return fig, ax, fig2, ax2

def q_1_3_1(N=1000, P=50, beta=20):
    f_errs = np.arange(0.05, 0.5, 0.05)
    print(f_errs)
    hn = HopfieldNetwork(N, beta=beta)
    patterns = hn.generate_random_patterns(P)
    _ = hn.train()
    print(patterns.shape)
    # choose 20 random patterns out of the memory patterns
    p_smpl = 20
    erroneous_patterns = rng.choice(patterns, p_smpl, replace=False)
    single_errs = np.zeros((f_errs.size, p_smpl))
    currect_recalls = np.zeros((f_errs.size, p_smpl))

    def sim_helper(p, N, f_err ):
        err_idxs = rng.choice(np.arange(N, dtype=int), size=(int(f_err * N)), replace=False)
        err_p = p.copy()
        err_p[err_idxs] = 1 - err_p[err_idxs]
        sol = hn.simulate(r0=err_p, t_span=(0, 20*hn.tau))
        return sol

    for i, f_err in enumerate(f_errs):
        for j, p in enumerate(erroneous_patterns):
            sol = sim_helper(p, N, f_err)
            single_errs[i,j] = single_err = hn.get_error(sol.y[:,-1], p)
            currect_recalls[i,j] = single_err <= 0.05
    
    fig, ax = plt.subplots()
    ax.plot(f_errs, single_errs.mean(axis=1), 'o-', label="Mean error rate")
    ax.plot(f_errs, currect_recalls.mean(axis=1), 'o-', label="Mean correct recalls")
    ax.set_xlabel("Fraction of errors")
    ax.set_ylabel("Mean error rate")
    fig.suptitle("Q1.3: Mean error rate and correct recalls as a function of the fraction of errors")
    ax.set_title(f"β={beta}")

    ax.legend()
    return fig, ax

def q_1_3(N=1000, P=50, beta=20):
    for beta in [20, 15, 10]:
        fig, ax = q_1_3_1(N, P, beta)
    
#%%
if __name__ == "__main__":
    P, N = 50, 1000
    
    start = time.time()
    q_1_1(N)
    end = time.time()
    print("Q1.1: Execution time:", end - start)
    plt.show()

    start = time.time()
    q_1_2(N, P)
    end = time.time()
    print("Q1.2: Execution time:", end - start)
    plt.show()

    start = time.time()
    q_1_3(N, P)
    end = time.time()
    print("Q1.3: Execution time:", end - start)
    plt.show()

