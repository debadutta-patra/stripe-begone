<<<<<<< HEAD
# Stripe Begone

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

## Usage

1. **Launch the Application**:
   ```bash
   python main_gui.py
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
=======
# stripe-begone

Simple application for removing stripe and curtaining artifacts from EM images.

## Installation

Precompiled version of the app is available for linux 64-bit and Windows platform. The precompiled version has been tested on Ubuntu 24.04 and Windows 10/11.

To directly run using the source code it is recommende to use a virtual environment. The app was created using pyhton 3.12 and ttkbootstrap, but it should work on older versions of python3.

### Create virtual environment

To create a virtual environment you can use either `venv` or `conda`.

With 'venv' use the following command

```bash:
python3 -m venv path/to/create/environment/folder
source path/to/created/environment/bin/activate
```

Alternatively if using conda use

```bash:
conda create -n myenvname python=3.12
conda activate myenvname
```

### Installing dependency

To install dependencies use 

```bash:
pip install -r requirements.txt
```

## Usage

You can start the application by double clicking the stripe-begone.exe or if using the source code type

```bash:
python main_gui.py
```

A brief description of how to use the software can be found inside the application under instructions tab.

## Contribution

This app was made by me as a weekend project. The wedge creation function was adapted from https://github.com/jtschwar/Removing-Stripe-Artifacts, but the image reconstruction algorithm here is much more quicker and simplistic than described in https://github.com/jtschwar/Removing-Stripe-Artifacts.

I hope this application was useful to you. if you have any concerns or comments please feel free to open an issue.
>>>>>>> 363d7d4046b81e76f9b24e19478ab330bb169a48
