# Continuous-Time Kalman–Bucy Filtering for 2D GPS Navigation with Nonlinear Extension

## Project Structure
```
project-root/
├── src/
│   ├── System.py       # State SDE simulation (Euler-Maruyama)
│   ├── GPSModel.py     # Observation SDE and Jacobian H(x̂t)
│   ├── KBF.py          # Kalman-Bucy filter and Riccati integration
│   └── EKBF.py         # Extended Kalman-Bucy filter
├── tests/              # Unit tests
├── results/            # Generated figures and outputs
├── demo.py             # Runs all three experiments and saves data
├── Experiment1.py      # Baseline run: well-spread satellites
├── Experiment2.py      # Degenerate geometry study
└── Experiment3.py      # Initialization sensitivity
```

## Installation

Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`
- `matplotlib`

## Usage

Run the main simulation to generate data for all three experiments:
```bash
python demo.py
```

Then run the corresponding experiment file to generate plots:
```bash
python experiment1.py  # Baseline run: well-spread satellites
python experiment2.py  # Degenerate geometry study
python experiment3.py  # Initialization sensitivity
```
