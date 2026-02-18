# IteraStripe

Stripe Begone is a GUI-based tool designed to remove stripe artifacts from microscopy images (specifically SEM/FIB-SEM) using Fourier space masking and advanced reconstruction algorithms.

## Features

- **Interactive Fourier Masking**: Define wedges to mask specific frequency components corresponding to stripes.
- **Automated Angle Detection**: Uses Radon transform variance analysis to automatically detect stripe orientation.
- **Advanced Reconstruction Methods**:
  - **FISTA** (Fast Iterative Shrinkage-Thresholding Algorithm)
  - **POCS** (Projection Onto Convex Sets)
  - **Total Variation (TV)** Denoising
  - **Weighted L1** Minimization
  - **KNN** Imputation
  - **Zero Fill**
- **Texture Recovery**: Restore non-stripe high-frequency details lost during reconstruction.
- **Contrast Enhancement**: Built-in tools for histogram adjustment.
- **GPU Acceleration**: Utilizes PyTorch for accelerated reconstruction if CUDA is available.

## Requirements

- Python 3.8+
- PySide6
- NumPy
- SciPy
- scikit-image
- scikit-learn
- Matplotlib
- PyTorch

## Installation

### Prerequisites

*   Python 3.14 or higher recommended (Though it is not tested it should run with earlier version of Python 3).
*   [uv](https://github.com/astral-sh/uv) (optional, but recommended for building).

### Installation 
#### Quick Start with uv
The fastest way to try pyTRACTnmr without installation is:

```bash
uvx IteraStripe
```

#### Using pip

```bash
pip install IteraStripe
```

#### From Source

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/debadutta-patra/IteraStripe
    cd pyTRACTnmr
    ```

2.  **Install the package:**

    Using `uv` (fastest):
    ```bash
    uv pip install .
    ```

    Using standard `pip`:
    ```bash
    pip install .
    ```

## Usage

1. **Launch the Application**:
   ```bash
   python src/IteraStripe/main.py
   ```

2. **Load an Image**:
   - Click `File -> Open Image` or drag and drop an image file onto the window.

3. **Configure Masks**:
   - Use the **Wedge Parameters** controls to define the stripe angle and width.
   - Click **Auto** to attempt automatic angle detection.
   - Click **Add Wedge** to register the mask.

4. **Remove Stripes**:
   - Select a method from the **Processing** panel (Default: FISTA Reconstruction).
   - Click **Remove Stripes**.

5. **Refine and Save**:
   - Use the tabs to view the Reconstruction, Difference map, or FFT.
   - Apply **Texture Recovery** if the result is too smooth.
   - Save the final image.

