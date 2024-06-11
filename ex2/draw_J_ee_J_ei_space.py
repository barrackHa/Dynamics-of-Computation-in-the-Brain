#%% 
import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 10, 1000)

# (0 < (1 - J_EE + J_EI)) && (0 < (h_E_0 - (J_EI * h_I_0)))

# OR

# (0 < (1 - J_EE + J_EI) < 0) && ((h_E_0 - (J_EI * h_I_0)) < 0)

# Define the inequality function
#%% 
def discrimenent_is_pos(x):
    return ((x**2) / 4)

def determinant_is_pos(x):
    return (x - 1)

# Generate x values
jee = np.linspace(0, 3, 400)
dis_is_pos = discrimenent_is_pos(jee)
det_is_pos = determinant_is_pos(jee)

print(dis_is_pos[:5])
#%% 
# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))
# Plot the line representing the boundary of the inequality
ax.plot(jee, dis_is_pos, 'r-', label=r'$J_{EI}=\frac{J_{EE}^{2}}{4}, \Delta>0$')

# draw a horizontal line at J_EI = 1 starting from the curve
ax.plot(jee[jee<=2], det_is_pos[jee<=2], 'g-', 
    label=r'$J_{EI}=J_{EE}-1$, Det(M)>0 '
)

# draw a vertical line at J_EE = 2 starting from the curve
ax.vlines(
    x=2, ymin=discrimenent_is_pos(2), ymax=dis_is_pos.max(), 
    color='black', linestyle='-', label=r'$J_{EE}=2$'
)

# Shade the region satisfying the inequality y >= 2x + 1
ax.fill_between(
    jee, dis_is_pos, 
    y2=np.max(dis_is_pos), where=(
        2 <= jee), 
    color='b', alpha=0.3, 
    label=r'$2<J_{EE}$ & $\frac{J_{EE}^{2}}{4}$<$J_{EI}$ - Complex Unstable'
)

ax.fill_between(
    jee, dis_is_pos, 
    y2=np.max(dis_is_pos), where=(
        2 > jee), 
    color='orange', alpha=0.3, 
    label=r'$J_{EE}<2$ & $\frac{J_{EE}^{2}}{4}$<$J_{EI}$ - Complex Stable'
)

ax.fill_between(
    jee, dis_is_pos, det_is_pos, 
    where=((dis_is_pos > det_is_pos) & (jee <= 2)), 
    color='green', alpha=0.3, label='Real Eignvalues, Stable Sol'
)

# Add labels and title
ax.set_xlabel(r'$J_{EE}$', rotation=0)
ax.set_ylabel(r'$J_{EI}$', rotation=0, labelpad=15)
# ax.set_title('Region for the Inequality y ≥ 2x + 1')

ax.text(2, 0.5, "Real Eignvalues, \nUnstable Solution", fontsize=14)
ax.text(2.1, 1.6, "Complex\nEignvalues, \nUnstable \nSolution", fontsize=14)
ax.text(0.5, 1.1, "Complex Eignvalues, \nStable Solution", fontsize=14)

# Add a legend
ax.legend()

# Add a grid
ax.grid(True)

# Set plot limits
ax.set_xlim([0, 3])
ax.set_ylim([0, discrimenent_is_pos(3)])

# Show the plot
plt.show()
# %%
