# Lab 07 – Evidence and Verification

## 1. Purpose

This folder contains evidence of the practical work completed in Lab 07 – Data Preprocessing for Deep Learning.

The evidence will demonstrate the Python environment, installed libraries, successful execution of the preprocessing script, and its output.

## 2. Evidence to Capture

### Evidence 01 – Python Environment and Library Versions

Capture a terminal screenshot showing:

- Python version.
- Active virtual environment.
- NumPy version.
- Scikit-learn version.

Verification command:

```bash
python3 --version
python3 -c "import numpy, sklearn; print(numpy.__version__, sklearn.__version__)"
```

### Evidence 02 – Data Normalization and Standardization

Capture the terminal output showing:

- Normalized sample.
- Standardized sample.
- Standardized column means.
- Standardized column standard deviations.

Run:

```bash
python3 python-practicals/preprocess.py
```

### Evidence 03 – Training and Testing Split

Capture the output showing:

```text
Training Features Shape: (70, 5)
Testing Features Shape: (30, 5)
```

### Evidence 04 – Scaling and Verification

Capture the output showing:

```text
Normalized train min/max: 0.0 1.0
Test normalized min/max: -0.002 1.055
Standardized train mean: 0.0
Standardized train std: 1.0
```

The test normalized values can be slightly outside 0 to 1 because the scaler was fitted only on training data.

## 3. Verification Checklist

- [x] Python practical created.
- [x] Seven practical steps implemented.
- [x] MinMaxScaler tested.
- [x] StandardScaler tested.
- [x] Training and testing split completed.
- [x] Data leakage prevention implemented.
- [x] Python practical executed successfully.
- [ ] Terminal screenshots captured and saved in this folder.

## 4. Conclusion

The practical execution was successful. The screenshots listed above will provide supporting evidence of the completed data preprocessing tasks.

**Note:** Save actual screenshots in this evidence folder after capturing them. Do not mark the screenshot checklist as complete until the files have been saved.
