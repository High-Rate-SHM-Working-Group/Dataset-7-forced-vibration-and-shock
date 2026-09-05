import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ------------------------------------------------------------------
# Style and formatting
# ------------------------------------------------------------------
plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif']})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})

# ------------------------------------------------------------------
# Path to the .lvm file
# ------------------------------------------------------------------
lvm_path = 'vibration_0.3.lvm'


def load_lvm(path):

    required_columns = [
        'time',
        'excitation_vibration',
        'excitation_shock',
        'excitation_composite',
        'accelerometer_base',
        'accelerometer_board'
    ]

    with open(path, 'r') as f:
        lines = f.readlines()

    # --------------------------------------------------------------
    # Locate the header
    # --------------------------------------------------------------
    header = None
    header_index = None

    for i, line in enumerate(lines):

        cols = [c.strip().lower() for c in line.strip().split(',')]

        if all(col in cols for col in required_columns):
            header = cols
            header_index = i
            break

    if header is None:
        raise ValueError("Could not locate the required data header.")

    # Map column names to indices
    column_map = {
        name: idx
        for idx, name in enumerate(header)
    }

    rows = []

    # --------------------------------------------------------------
    # Read data
    # --------------------------------------------------------------
    for line in lines[header_index + 1:]:

        if not line.strip():
            continue

        parts = [p.strip() for p in line.rstrip().split(',')]

        # Pad missing trailing columns (e.g., empty Comment)
        while len(parts) < len(header):
            parts.append('')

        # Ignore lines where the time column isn't numeric
        try:
            float(parts[column_map['time']])
        except (ValueError, IndexError):
            continue

        rows.append(parts)

    if len(rows) == 0:
        raise ValueError("No valid data rows found.")

    # --------------------------------------------------------------
    # Convert strings to floats
    # --------------------------------------------------------------
    def to_float(x):
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan

    data = {}

    for name in required_columns:
        idx = column_map[name]
        data[name] = np.array(
            [to_float(row[idx]) for row in rows]
        )

    # --------------------------------------------------------------
    # Remove rows with invalid time
    # --------------------------------------------------------------
    mask = ~np.isnan(data['time'])

    for key in data:
        data[key] = data[key][mask]

    # Equal lengths
    N = min(len(v) for v in data.values())

    for key in data:
        data[key] = data[key][:N]

    return (
        data['time'],
        data['excitation_vibration'],
        data['excitation_shock'],
        data['excitation_composite'],
        data['accelerometer_base'],
        data['accelerometer_board']
    )


# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
time, vib, shock, comp, base, board = load_lvm(lvm_path)

print(f'Trimmed length: {len(time)} samples')
print(f'Time range: {time[0]:.6f} to {time[-1]:.6f} s')

# ------------------------------------------------------------------
# Plot
# ------------------------------------------------------------------

signals = [
    (vib,   r'excitation\_vibration'),
    (shock, r'excitation\_shock'),
    (comp,  r'excitation\_composite'),
    (base,  r'accelerometer\_base'),
    (board, r'accelerometer\_board')
]

# --------------------------------------------------------------
# Edit plot settings here
# --------------------------------------------------------------
plot_settings = [
    {
        "xlim": (0, 7.8),
        "ylim": (-0.45, 0.45),
        "xticks": np.arange(0, 8, 1),
        "yticks": np.arange(-0.4, 0.5, 0.2),
    },
    {
        "xlim": (0, 7.8),
        "ylim": (-0.10, 1.10),
        "xticks": np.arange(0, 8, 1),
        "yticks": np.arange(0, 1.1, 0.2),
    },
    {
        "xlim": (0, 7.8),
        "ylim": (-0.50, 1.20),
        "xticks": np.arange(0, 8, 1),
        "yticks": np.arange(-0.4, 1.3, 0.4),
    },
    {
        "xlim": (0, 7.8),
        "ylim": (-1.20, 1.20),
        "xticks": np.arange(0, 8, 1),
        "yticks": np.arange(-1.0, 1.1, 0.5),
    },
    {
        "xlim": (0, 7.8),
        "ylim": (-1.20, 1.20),
        "xticks": np.arange(0, 8, 1),
        "yticks": np.arange(-1.0, 1.1, 0.5),
    }
]

for (sig, ylabel), ps in zip(signals, plot_settings):

    fig, ax = plt.subplots(figsize=(4, 2.0))

    ax.plot(time, sig, linewidth=0.8)

    ax.set_ylabel(ylabel)

    ax.grid(True, which='both', linestyle=':', alpha=0.6)

    ax.set_xlim(ps["xlim"])
    ax.set_ylim(ps["ylim"])

    ax.set_xticks(ps["xticks"])
    ax.set_yticks(ps["yticks"])

    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    ax.tick_params(axis='both', which='both', direction='out')

    ax.set_xlabel(r'time (s)')

    ax.set_title(ylabel, pad=8)

    fig.tight_layout()
    plt.show()
