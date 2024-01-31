import matplotlib.pyplot as plt
import numpy as np

def debug(function_name):
    fn_name = np.genfromtxt(f'{function_name}.dat', names = True)
    print(fn_name['n'])
    print(fn_name[f'my_{function_name}_time'])
    print(fn_name[f'cublas_{function_name}_time'])

def savePlot(function_name):
    fn_name = np.genfromtxt(f'{function_name}.dat', names = True)
    x = range(len(fn_name['n']))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(function_name)
    fig.set_size_inches(18.5, 10.5, forward=True)
    fig.set_dpi(100)

    ax1.plot(x, fn_name[f'my_{function_name}_time'], marker = 'o', label = f'my {function_name} time')
    ax1.plot(x, fn_name[f'cublas_{function_name}_time'], marker = 's', label = f'cublas {function_name} time')
    ax1.set_xlabel('n')
    ax1.set_ylabel('ms')
    ax1.legend()
    ax1.set_title(f'Runtime Comparison', pad = 4)
    ax1.tick_params(labelrotation = 60)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fn_name['n'])

    ax2.axhline(0, color = 'red', ls = '--')
    ax2.bar(x, fn_name[f'cublas_{function_name}_time'] - fn_name[f'my_{function_name}_time'], label = f'{function_name} time difference')
    ax2.set_xlabel('n')
    ax2.set_ylabel('ms')
    ax2.legend()
    ax2.set_title(f'Runtime Comparison (cuBlas - my{function_name})', pad = 4)
    ax2.tick_params(labelrotation = 60)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fn_name['n'])

    fig.tight_layout(pad=0.4)
    fig.savefig(f'../misc/{function_name}.png')
    plt.clf()

function_names = ['dasum', 'dnrm2', 'dgemm', 'daxpy', 'dcopy']

for function_name in function_names:
    # debug(function_name)
    savePlot(function_name)
