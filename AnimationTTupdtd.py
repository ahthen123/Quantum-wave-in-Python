# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:35:14 2026

@author: James Tan SH
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dx = 0.05
dt = 0.02
c = 1.0
l = 10.0
x = np.arange(0, l, dx)
nx = len(x)

# Courant stability parameter (must be <= 1 for stability)
r_sq = (c * dt / dx) ** 2

# Initial Gaussian wave packet centred at x = 5
u_prev = np.exp(-((x - 5.0) ** 2) / 0.5)
u_curr = np.copy(u_prev)
u_next = np.zeros(nx)

# Figure setup
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, l)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('Spatial Position x (m)', fontsize=10)
ax.set_ylabel('Wave Amplitude u(x)', fontsize=10)
ax.set_title('Live 1D Wave Packet Propagation Simulation', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)

line, = ax.plot(x, u_curr, color='#1f77b4', lw=2)

def update(frame):
    global u_prev, u_curr, u_next

    # Finite-difference update (interior nodes only)
    u_next[1:-1] = (2 * u_curr[1:-1] - u_prev[1:-1]
                    + r_sq * (u_curr[2:] - 2 * u_curr[1:-1] + u_curr[:-2]))

    # Fixed boundaries
    u_next[0] = 0
    u_next[-1] = 0

    # Advance time levels
    u_prev = np.copy(u_curr)
    u_curr = np.copy(u_next)

    line.set_ydata(u_curr)
    return line,

ani = animation.FuncAnimation(fig, update, frames=200, interval=20, blit=True)
plt.show()
