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
