import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------------
# Style and formatting
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})
# ------------------------------------------------------------------

# ---------- Path to the .lvm file ----------
lvm_path = 'run_4.lvm'      # Change this for each file
# -------------------------------------------


def load_lvm(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Locate the data header
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith(',time,excitation_vibration'):
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f'Could not find the data header in {path}')

    rows = []
    for line in lines[data_start:]:
        parts = [p.strip() for p in line.split(',')]

        # Ignore the empty first column
        if len(parts) >= 7:
            rows.append(parts[1:7])

    def to_float(x):
        try:
            return float(x) if x else np.nan
        except ValueError:
            return np.nan

    data = np.array([[to_float(c) for c in r] for r in rows])

    # Columns
    time  = data[:, 0]
    vib   = data[:, 1]
    shock = data[:, 2]
    comp  = data[:, 3]
    base  = data[:, 4]
    board = data[:, 5]

    # Remove invalid timestamps
    mask = ~np.isnan(time)
    time  = time[mask]
    vib   = vib[mask]
    shock = shock[mask]
    comp  = comp[mask]
    base  = base[mask]
    board = board[mask]

    # Make all channels the same length
    N = min(len(time), len(vib), len(shock),
            len(comp), len(base), len(board))

    return (
        time[:N],
        vib[:N],
        shock[:N],
        comp[:N],
        base[:N],
        board[:N]
    )


# Load data
time, vib, shock, comp, base, board = load_lvm(lvm_path)

print(f'Trimmed length: {len(time)} samples')
print(f'Time range: {time[0]:.6f} to {time[-1]:.6f} s')

# -------------------------------------------------
# Plot
# -------------------------------------------------

fig, axes = plt.subplots(
    5, 1,
    figsize=(10, 12),
    sharex=True
)

signals = [
    (vib,   r'excitation_vibration'),
    (shock, r'excitation_shock'),
    (comp,  r'cxcitation_composite'),
    (base,  r'accelerometer_base'),
    (board, r'accelerometer_board')
]

for ax, (sig, ylabel) in zip(axes, signals):

    ax.plot(time, sig, linewidth=0.8)

    ax.set_ylabel(ylabel)

    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    # Show x-axis tick labels on EVERY subplot
    ax.tick_params(axis='x', which='both', labelbottom=True)

    # Set the x-axis limits
    ax.set_xlim(0, 7.8)
    
    # Set the x-axis tick locations manually
    ax.set_xticks(np.arange(0, 8, 1))
    
    # Optional: format tick labels as plain numbers
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

    # Optional: label every subplot
    ax.set_xlabel(r'time (s)')

fig.suptitle(
    r'all channels vs. time',
    y=0.995
)

fig.tight_layout(rect=[0, 0.03, 1, 0.98])

plt.show()