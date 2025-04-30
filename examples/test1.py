import wrmodes as wr
import scipy.constants as sc


# WR-90 sides in meters
wr90_long = 0.9 * 0.0254
wr90_short = 0.4 * 0.0254

# Maximum frequency in Hz
max_frequency = 50e9  # 50 GHz

# List all propagating modes in WR-90 at frequency `max_frequency`
modes = wr.list_propagating_modes(
    max_frequency,
    wr90_long, 
    wr90_short, 
    mu=sc.mu_0,
    epsilon=sc.epsilon_0,
)

for mode in modes:
    print(f"Mode: {mode[0]}, Cutoff Frequency: {mode[3]*1e-9:.2f} Hz")