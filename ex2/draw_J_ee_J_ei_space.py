import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 10, 1000)

# (0 < (1 - J_EE + J_EI)) && (0 < (h_E_0 - (J_EI * h_I_0)))

# OR

# (0 < (1 - J_EE + J_EI) < 0) && ((h_E_0 - (J_EI * h_I_0)) < 0)

# Define the inequality function
def inequality_line(x):
    return 2 * x + 1

# Generate x values
x = np.linspace(-5, 5, 400)
y = inequality_line(x)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the line representing the boundary of the inequality
ax.plot(x, y, 'r-', label='y = 2x + 1')

# Shade the region satisfying the inequality y >= 2x + 1
ax.fill_between(x, y, y2=np.max(y) + 10, where=(y <= np.max(y) + 10), color='gray', alpha=0.3, label='y ≥ 2x + 1')

# Add labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Region for the Inequality y ≥ 2x + 1')

# Add a legend
ax.legend()

# Add a grid
ax.grid(True)

# Set plot limits
ax.set_xlim([-5, 5])
ax.set_ylim([-10, 20])

# Show the plot
plt.show()