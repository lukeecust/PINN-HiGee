import numpy as np
from numpy import pi, sqrt
from scipy.interpolate import interp1d
from fipy import CellVariable, Grid1D, TransientTerm, DiffusionTerm
from fipy.solvers import LinearLUSolver

# -----------------------------
# Experimental conditions
# -----------------------------
pH = 12.5                               # Alkaline H2O2 pH
ch = 1                                  # H2O2 concentration
T = 298.15                              # Temperature (K)
Pno = 101.325                           # NO partial pressure at the interface (kPa)
Hno = 709.1725                          # Henry constant / solubility coefficient
cno = Pno / Hno                         # Equilibrium dissolved NO concentration (mol/L)
ooh = 1.0 / (1.0 + 10 ** (11.73 - pH)) * ch   # Bulk HOO- from H2O2 pKa (mol/L)

# -----------------------------
# De vs rpm (interpolation)
# -----------------------------
rpm_data = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])
De_data = np.array([5.3064, 6.2257, 6.7727, 7.1618, 7.4629, 7.7081, 7.9144, 8.0818])
f_De = interp1d(rpm_data, De_data, kind="linear")
def get_de(rpm):
    """Return interpolated De for a given rpm."""
    if rpm < rpm_data.min() or rpm > rpm_data.max():
        raise ValueError("rpm is out of the interpolation range (extrapolation/data needed).")
    return f_De(rpm)

# -----------------------------
# RPB parameters
# -----------------------------
rpm = 1200
omega = 2 * pi * rpm / 60.0             # Angular velocity (rad/s)
De = get_de(rpm)
r1 = 26e-3                              # Inner radius (m)
r2 = 70e-3                              # Outer radius (m)
r = sqrt((r1 ** 2 + r2 ** 2) / 2)       # Mean radius (m)
h = 25e-3                               # Bed height (m)
ap = 550                                # Specific surface area (m^2/m^3)
V = pi * (r2 ** 2 - r1 ** 2) * h        # Reactor volume (m^3)
# -----------------------------
# Gas/liquid flowrates
# -----------------------------
Q = 20e-3 / 3600                        # Liquid flowrate (m^3/s) [20 L/h]
gn2 = 2000e-3 / 3600                    # N2 flowrate (m^3/s) [2000 L/h]
r0 = 5e-4                               # Nozzle radius (m)
aa = pi * (r0 ** 2)                     # Nozzle area (m^2)

u0 = 0.02107 * Q ** 0.2279 * (omega ** 2 * r)  # Mean radial velocity (m/s)
g0 = 100
v0 = 1.0e-6                             # Liquid viscosity
porosity = 0.97

# -----------------------------
# Derived quantities: ts, holdup, d, a_bi
# -----------------------------
ts = (r2 - r1) / (u0 * 31.0)            # Film renewal time (s)
holdup = (0.039 * ((r * omega ** 2 / g0) ** (-0.5)) * ((u0 / 0.01) ** 0.6) * ((1.35e-6 / v0) ** 0.22))
d = (12.84 * ((72.0e-2) / (r * omega ** 2 * 1000.0)) ** 0.630 * ((Q * 3600) ** 0.201))
x_liquid = d / 2                        # Liquid film thickness (m)
a_bi = 6.0 * holdup / d                 # Interfacial area (1/m)

# -----------------------------
# Print summary
# -----------------------------
print("==== Derived quantities ====")
print(f"pH = {pH}")
print(f"cNO = {cno}")
print(f"ooh (HOO- concentration) = {ooh}")
print(f"Radial velocity u0 = {u0} m/s")
print(f"Film renewal time ts = {ts} s")
print(f"Holdup = {holdup}")
print(f"Mean radius r = {r}")
print(f"Droplet diameter d = {d}")
print(f"Interfacial area a_bi = {a_bi}")
print(f"Effective diffusion factor De at rpm={rpm}: {De}")

# Kinetics and diffusivities
k = 103.4                               # Rate constant (m^3/(mol*s))
D1 = 2.21e-9 * De                       # Diffusivity for NO (m^2/s)
D2 = 1.97e-9 * De                       # Diffusivity for HOO- (m^2/s)

# Grid settings
x_max = x_liquid
time_max = ts
nx = 800
n_t = 1000

mesh = Grid1D(dx=x_max / nx, nx=nx)
c1 = CellVariable(name="NO concentration (c1)", mesh=mesh, value=0.0, hasOld=True)
c2 = CellVariable(name="OOH concentration (c2)", mesh=mesh, value=0.0, hasOld=True)

# PDE system
eq1 = TransientTerm(var=c1) == DiffusionTerm(coeff=D1, var=c1) - k * c1 * c2
eq2 = TransientTerm(var=c2) == DiffusionTerm(coeff=D2, var=c2) - k * c1 * c2
eq_coupled = eq1 & eq2

# Boundary conditions
c1.constrain(cno, where=mesh.facesLeft)
c2.constrain(ooh, where=mesh.facesRight)

# Solver configuration
solver = LinearLUSolver(iterations=1000, tolerance=1e-8)

# Time marching
dt = time_max / n_t
nt = int(time_max / dt)

c1_results = np.zeros((nt + 1, nx), dtype=float)
c2_results = np.zeros((nt + 1, nx), dtype=float)
c1_results[0, :] = c1.value
c2_results[0, :] = c2.value

max_sweeps = 20
tolerance = 1e-6

for step in range(nt):
    c1.updateOld()
    c2.updateOld()

    for sweep in range(max_sweeps):
        residual = eq_coupled.sweep(dt=dt, solver=solver)
        if residual < tolerance:
            break
    if sweep == max_sweeps - 1 and residual > tolerance:
        print(f"Warning: step {step + 1} did not converge (residual={residual:.2e}).")
    c1_results[step + 1, :] = c1.value
    c2_results[step + 1, :] = c2.value
    if (step + 1) % 100 == 0:
        print(f"Completed {step + 1}/{nt}, last residual={residual:.2e}")
print("Calculation finished.")

# Save results
np.savetxt("c1_results.csv", c1_results, delimiter=",", fmt="%.6e")
np.savetxt("c2_results.csv", c2_results, delimiter=",", fmt="%.6e")
