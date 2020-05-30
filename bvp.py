import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sympy import *
dtype = np.float32

def bvp(kind: int, x_0: float, y_0: float, x_n: float, y_n: float, F: function, G: function):
    x = Symbol('x')
    y = Function('y')(x)
    F, G = F(x), G(x)
    ode = Eq(
        y.diff(x, 2) + F * y,
        G,
    )
    dy = y.diff(x, kind - 1)
    ics = {
        dy.subs(x, x_0): y_0,
        dy.subs(x, x_n): y_n,
    }
    Y = dsolve(ode, y, ics=ics).rhs.evalf()
    return {'x': x, 'y': y}, {'Y': Y, 'F': F, 'G': G}

# xlim = {'x_0': 0, 'x_n': 15}
# conditions = {
#     'kind': 2,
#     'y_0': 1, 'y_n': 1,
#     'F': lambda x: -1 + 2 * x,
#     # 'F': lambda x: 10 / (1 + x),
#     'G': lambda x: 0 * x,
# }
# conditions.update(xlim)
# variables, functions = bvp(**conditions)
# functions['Y'].subs(variables['x'], 2).evalf()

def to_arrays(xlim: dict, x: Symbol, y: Function, Y: Function, F: Function, G: Function):
    x_0, x_n = xlim['x_0'], xlim['x_n']
    n = (x_n - x_0) * 10
    aX = np.linspace(x_0, x_n, n, dtype=dtype)
    aY = np.array([Y.subs(x, i).evalf() for i in aX], dtype=dtype)
    aF = np.array([F.subs(x, i).evalf() for i in aX], dtype=dtype)
    # aG = np.array([G.subs(x, i).evalf() for i in aX], dtype=dtype)
    arrays = {'Y(x)': np.array([aX, aY]), 'F(x)': np.array([aX, aF])}#, 'G(x)': np.array([aX, aG])}
    return arrays

def plot(xlim: dict, arrays: dict):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.gca()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([xlim['x_0'], xlim['x_n']])
    ax.grid()
    for label, value in arrays.items():
        ax.plot(*value, label=label)
    ax.legend()
    return fig

# arrays = to_arrays(xlim, **variables, **functions)
# # del arrays['G(x)']
# fig = plot(xlim, arrays)

from multiprocessing import Pool
from tqdm import tqdm

def iter_samples():
    # np.random.seed(0)
    n_samples = 2000
    B = np.linspace(-np.pi / 2, np.pi / 2, n_samples)
    for i in tqdm(range(n_samples)):
        t = i % 10
        xlim = {'x_0': t, 'x_n': t + 10}
        conditions = {
            'kind': 1,
            'y_0': 1, 'y_n': 2,
        }
        conditions.update(xlim)
        yield i, B, conditions, xlim

def save_sample(args):
    i, B, conditions, xlim = args
    conditions['F'] = lambda x: B[i] - B[i] * x
    variables, functions = bvp(**conditions)
    # print(f'{i} - solved:', functions['Y'].subs(variables['x'], 2).evalf())
    arrays = to_arrays(xlim, **variables, **functions)
    inputs = arrays['Y(x)'][1]
    targets = arrays['F(x)'][1]
    samples = np.vstack((inputs, targets))
    np.save(f'dataset/{i}.npy', samples)
    fig = plot(xlim, arrays)
    fig.savefig(f'trains/{i}.png')
    plt.close()
    print(f'{i} - saved!')

if __name__ == '__main__':
    with Pool(4) as pool:
        pool.map(save_sample, iter_samples())
