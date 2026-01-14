import wrmodes as wr
import numpy as np
import matplotlib.pyplot as plt
import os


a = 44e-3
b = 30e-3
theta_deg = 90
f = 9.5e9

te10_s21 = []
te20_s21 = []

radii = np.arange(30e-3, 260e-3, 5e-3)

for R0 in radii:

    def c_radius(z):
        return 1 / (R0 * np.ones_like(z))
    
    theta = np.radians(theta_deg)
    LB = R0 * theta

    mode_in = ('TE', 1, 0)

    coupled_modes, result = wr.solve_mode_coupling(f, a, b, mode_in, length=LB, curvature=c_radius, verbose=False, maximum_modes=11, resolution='auto', tol=1e-1)

    # Calculate S21
    i = [m[0] for m in coupled_modes].index(f"TE1,0")
    j = [m[0] for m in coupled_modes].index(f"TE2,0")

    te10_s21.append(np.abs(result.y[i, -1])**2)
    te20_s21.append(np.abs(result.y[j, -1])**2)


plt.figure(figsize=(4, 3))
plt.plot(radii*1e3, te10_s21, label=r'TE$_{1,0}$')
plt.xlabel("Curvature radius (mm)")
plt.ylim(0.6, 1)
plt.ylabel('Normalized transmission efficiency')
plt.legend(loc=(0, 1), frameon=False)
plt.twinx()
plt.plot(radii*1e3, te20_s21, label=r'TE$_{2,0}$', color='C1')
plt.ylim(0, 0.4)

plt.tight_layout()
plt.xlim(30, 250)
plt.legend(loc=(0.8, 1), frameon=False)

plt.savefig("test2_te10_te20_s21_vs_curvature_radius.png", dpi=300)

# Tell user where the figure was saved
print(f"Figure saved in: {os.getcwd()}/test2_te10_te20_s21_vs_curvature_radius.png")

plt.show()