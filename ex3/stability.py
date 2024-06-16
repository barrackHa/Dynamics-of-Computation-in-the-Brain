#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
x = np.linspace(-1.5, 5, 100)
y = np.sin(x)
circ_size = 100 


fig, ax = plt.subplots()
ax.plot(x, y, linewidth=4, label='sin(x)')


# Stable fixed points
ax.scatter(
    0, 0, facecolors='none', edgecolors='black', 
    label='Unstable FP (0,0)', s=circ_size
)

ax.arrow( -0.1, 0, -0.8, 0, head_width = 0.1, color='r', linewidth=2)
ax.arrow( 0.1, 0, 0.8, 0, head_width = 0.1, color='g', linewidth=2)

# unstable fixed points
ax.scatter(
    np.pi, 0, facecolors='black', edgecolors='black', 
    label='Stable FP ($\pi$,0)', s=circ_size
)

ax.arrow(1, 0, 1.9, 0, head_width = 0.1, color='g', linewidth=2)
ax.arrow(np.pi+1.1, 0, -0.8, 0, head_width = 0.1, color='r', linewidth=2)

ax.grid()
ax.set_facecolor('lightgray')
ax.set_title('Sin(x)')
ax.set_xlabel('x')
ax.set_ylabel(r'$\dot{x}$', rotation=0)
ax.legend()
plt.show()
# %%




# %%
x = np.linspace(-1.5, 5, 100)
y = -1 * np.sin(x)
circ_size = 100 


fig1, ax1 = plt.subplots()
ax1.plot(x, y, linewidth=4, label='sin(x)')

# UnStable fixed points
ax1.scatter(
    0, 0, facecolors='black', edgecolors='black', 
    label='Stable FP (0,0)', s=circ_size
)

ax1.arrow( 2.5, 0, -2.2, 0, head_width = 0.1, color='r', linewidth=2)
ax1.arrow( -1.1, 0, 0.8, 0, head_width = 0.1, color='g', linewidth=2)

# unstable fixed points
ax1.scatter(
    np.pi, 0, facecolors='none', edgecolors='black', 
    label='UnStable FP ($\pi$,0)', s=circ_size
)

ax1.arrow(np.pi+0.1, 0, 0.8, 0, head_width = 0.1, color='g', linewidth=2)
ax1.arrow(np.pi-0.1, 0, -0.8, 0, head_width = 0.1, color='r', linewidth=2)

ax1.grid()
ax1.set_facecolor('lightgray')
ax1.set_title('-Sin(x)')
ax1.set_xlabel('x')
ax1.set_ylabel(r'$\dot{x}$', rotation=0)
ax1.legend()
plt.show()

# %%
