import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.animation as animation
import glob, re

import scienceplots
plt.style.use('default')
plt.style.use(['science', 'high-vis'])

if len(sys.argv) < 2:
    print("Pass the file as argument")
file = sys.argv[1]

Acceptor, LengthIntrinsic, BV, BrP, DepletionWidth = np.loadtxt(file, delimiter=',', unpack=True, skiprows=1)

# Heat map of BV
Acceptors = np.unique(Acceptor)
LengthIntrinsics = np.unique(LengthIntrinsic)
BV = BV.reshape(len(Acceptors), len(LengthIntrinsics))
BrP = BrP.reshape(len(Acceptors), len(LengthIntrinsics))
DepletionWidth = DepletionWidth.reshape(len(Acceptors), len(LengthIntrinsics))
# Transpose
BV = BV.T
BrP = BrP.T
DepletionWidth = DepletionWidth.T
STEP = len(Acceptors) // 4

extent = [Acceptors[0], Acceptors[-1],
          LengthIntrinsics[0], LengthIntrinsics[-1]]
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(BV, extent=extent, aspect='auto', origin='lower',
               cmap='jet', interpolation="bicubic")
ax.set_xscale('log')
ax.set_xlabel('Acceptor')
ax.set_ylabel('LengthIntrinsic')
ax.set_title('BV')
fig.colorbar(im, ax=ax, label='BV')
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(BrP, extent=extent, aspect='auto', origin='lower', cmap='jet', interpolation="bicubic")
ax.set_xscale('log')
ax.set_xlabel('Acceptor')
ax.set_ylabel('LengthIntrinsic')
ax.set_title('BrP')
fig.colorbar(im, ax=ax, label='BrP')
plt.show()

# DepletionWidth figure
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(DepletionWidth, extent=extent,
               aspect='auto', origin='lower', cmap='jet', interpolation="bicubic")
ax.set_xscale('log')
ax.set_xlabel('Acceptor')
ax.set_ylabel('LengthIntrinsic')
ax.set_title('DepletionWidth')
fig.colorbar(im, ax=ax, label='DepletionWidth')
plt.show()
