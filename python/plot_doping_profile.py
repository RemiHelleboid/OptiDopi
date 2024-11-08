import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sys ,os
import glob


import scienceplots
plt.style.use('default')
plt.style.use(['science', 'high-vis', 'grid'])


def import_data(filename):
    data = np.loadtxt(filename, skiprows=1, delimiter=',')
    return data

def plot_doping_profile(filename):
    data = import_data(filename)
    # plt.plot(data[:,0], data[:,1], label='Donor')
    # plt.plot(data[:,0], data[:,2], label='Acceptor')
    plt.plot(data[:,0], np.abs(data[:,3]), label='Total')
    plt.xlabel('x ($\mu$m)')
    plt.ylabel('Doping (cm-3)')
    plt.yscale('log')
    plt.show()

def animation_deoping(dirname):
    file_list = glob.glob(dirname + '/doping*.csv')
    file_list.sort()
    print(f"No. of files: {len(file_list)}")
    
    list_X, listY = [], []
    listD, listA = [], []
    for filename in file_list:
        data = import_data(filename)
        list_X.append(data[:,0])
        listY.append(np.abs(data[:,3]))
        listD.append(data[:,1])
        listA.append(data[:,2])
    
    fig, ax = plt.subplots()
    ax.set_xlabel('x ($\mu$m)')
    ax.set_ylabel('Doping (cm-3)')
    ax.set_yscale('log')
    ax.set_xlim(0, 10)
    ax.set_ylim(1e11, 1e20)
    ax.set_title('Iteration: 0')

    line, = ax.plot(list_X[0], listY[0], lw=2, c='k', ls='-', zorder=100)
    line2, = ax.plot(list_X[0], listD[0], lw=4, c='b', ls='-')
    line3, = ax.plot(list_X[0], listA[0], lw=4, c='r', ls='-')
    s = ax.set_title(f'Iteration: 0')

    fig.tight_layout()

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        s = ax.set_title(f'Iteration: 0')
        return line, s, line2, line3
    
    def animate(i):
        line.set_data(list_X[i], listY[i])
        line2.set_data(list_X[i], listD[i])
        line3.set_data(list_X[i], listA[i])
        ax.set_title(f'Iteration: {i}')
        return line, ax, line2, line3
    
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(file_list), interval=100, blit=False, repeat=False)
    plt.show()
    anim.save('anim_doping_profile.mp4', fps=10, extra_args=['-vcodec', 'libx264'], dpi=300)
    

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python plot_doping_profile.py <filename>')
        sys.exit(1)
    arg_filename = sys.argv[1]
    # Check if its a file or a directory
    is_file = os.path.isfile(arg_filename)
    if is_file:
        plot_doping_profile(arg_filename)
    else:
        animation_deoping(arg_filename)

