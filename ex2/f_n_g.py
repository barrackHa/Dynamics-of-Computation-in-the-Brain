import numpy as np
import matplotlib.pyplot as plt

def f(x):
    if ((-1 * np.pi) <= x < (-0.5 * np.pi)):
        return -1
    elif ((-0.5 * np.pi) < x < (0.5 * np.pi)):
        return 1
    else:
        return -1
    
def g(x):
    if ((-1 * np.pi) <= x < 0):
        return -1
    else:
        return 1
    

if __name__ == "__main__":
    x = np.linspace(-1*np.pi, np.pi, 1000)
    y_f = [f(i) for i in x]
    z_g = [g(i) for i in x]
    
    fig, axs = plt.subplots(2, sharex=True)
    
    axs[0].plot(x, y_f, c='blue')
    axs[0].plot((x+np.pi*2), y_f, c='blue', linestyle='--')
    axs[0].plot((x+np.pi*-2), y_f, c='blue', linestyle='--')

    axs[1].plot(x, z_g, c='orange')
    axs[1].plot((x+np.pi*2), z_g, c='orange', linestyle='--')
    axs[1].plot((x+np.pi*-2), z_g, c='orange', linestyle='--')
    
    axs[0].set_title(r'$\bar{f}(x)$')
    axs[1].set_title(r'$\bar{g}(x)$')

    # set axs background to gray
    [ax.set_facecolor('lightgray') for ax in axs]
    
    [ax.grid() for ax in axs]
    plt.show()
    