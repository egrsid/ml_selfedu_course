from typing import List, Tuple
from matplotlib.lines import Line2D
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

FloatArray = NDArray[np.float64]

def f(x: np.float64):
    return -.5 * x + .2 * x ** 2 - .01 * x ** 3 - .3 * np.sin(4*x)

def df(x: np.float64):
    return -.5 + .4 * x - .03 * x**2 - 1.2 * np.cos(4*x)

N = 1000

coord_x = np.linspace(-5.0, 20.0, 400)
coord_y = f(coord_x)

plt.ion()
fig, ax = plt.subplots(figsize=(7,4))
ax.plot(coord_x, coord_y, color='tab:green', label='f(x)')

ax.set_xlim(coord_x.min(), coord_x.max())
ax.set_ylim(coord_y.min()-1, coord_y.max()+1)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(which='major', ls='--', lw=.6)
ax.grid(which='minor', ls=':',  lw=.4, alpha=.5)
ax.tick_params(which='minor', length=4)
ax.legend()

def update_plot(point: Line2D, x_curr):
    point.set_data([x_curr], [f(x_curr)])
    fig.canvas.draw_idle()
    plt.pause(0.001)

def start_anim(point3ar: List[Tuple[Line2D, np.float64, np.float64]])->None:
    x = np.full(len(point3ar), -5.0)
    v = np.zeros(len(point3ar))

    for _ in range(1, N + 1):
        for i, (point, eta, gamma) in enumerate(point3ar):
            update_plot(point, x[i])
            v[i] = gamma * v[i] + (1 - gamma) * eta * df(x[i])
            x[i] = x[i] - v[i]

(point1,) = ax.plot([], [], 'o', color="tab:orange")
(point2,) = ax.plot([], [], 'o', color="tab:purple")
(point3,) = ax.plot([], [], 'o', color="tab:cyan")

start_anim([
    (point1, .1, .8),
    (point2, .4, .8),
    (point3, .1, .95),
])

plt.ioff()
plt.show(block=True)