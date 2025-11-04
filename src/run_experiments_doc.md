# QCNN Automated Experiments

run_experimnets helps the user to automatically evauate given circuits with many different hyperparameters by running a training with each configuration and testing

## Quick Start

```bash
# Run all 20 experiments (~25-30 hours)
python run_experiments.py

# Each run creates a timestamped folder:
# output/run_20251015_100210/

# Check progress every ~8 hours
notepad output/run_YYYYMMDD_HHMMSS/run_log.txt

# After completion: view results
notepad output/run_YYYYMMDD_HHMMSS/final_results_table.txt

# Generate plots (update path to your run folder)
python analyze_results.py
```

---

## What It Does

Runs experiments with the given settings and given circuit, all possible combinations of the settings


**Check these in your run folder:**
- `run_log.txt` - Progress updates; minimal
- `final_results_table.txt` - Complete results table
- `experiment_analysis.png` - Comparison plots (after analyze_results.py)


## Analyzing Results

**After experiments complete:**

```bash
python analyze_results.py
```

**Automatically generates:**
- `experiment_analysis.png` - Multi-panel comparison plots
- `experiment_results.csv` - Spreadsheet-friendly export


**The script auto-detects the latest run** - no path needed

---

## Configuration

**An example configuration:**

Edit `run_experiments.py` to change:
```python
SEEDS = [1111, 2222, 3333, 4444, 5555]
OPTIMIZERS = ['COBYLA', 'SPSA']  # COBYLA runs first
EPOCHS = 300
NUM_TRAIN = 100
NUM_VAL = 75          # Extended validation size
NUM_VAL_QUICK = 15    # Quick validation size (every epoch)
NUM_VAL_EXTENDED_INTERVAL = 5  # Extended validation frequency
NUM_TEST = 250
EXPERIMENT_ORDER = 'amplitude_first' # Select what to get done first
```

**Validation Strategy:**
- **Quick validation** (15 images): Runs every epoch for progress plots
- **Extended validation** (75 images): Runs every 5 epochs for best model selection
- Tracks two best models: best training loss + best extended validation

**Start with `python run_experiments.py` and check `output/run_*/run_log.txt`.**
