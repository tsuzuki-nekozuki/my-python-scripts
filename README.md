# My Python scripts

## Introduction

This repository contains a collection of small Python scripts, utilities, and experiments that I use frequently.  
They are designed to be simple, reusable, and easy to understand.


## Environment

These scripts are tested in the following environment:

- **OS**: Arch Linux (Intel 64-bit)
- **Python**: 3.13.5

### Key Python Packages
| Package        | Version |
|----------------|---------|
| numpy          | 2.3.2   |
| pandas         | 2.3.1   |
| matplotlib     | 3.10.3  |
| scipy          | 1.16.1  |
| opencv-python  | 4.12.0  |
| openpyxl       | 3.1.5   |

### Hardware
- **CPU**: AMD Ryzen 7 5700X
- **Memory**: 32 GB
- **GPU**: NVIDIA RTX 3060 (12GB) with CUDA 12.9


## Scripts List

### Utils

- [Check environment](utils/system_info/check_system_info.py)
    Print system environment.

### Analysis

- [Convert matplotlib canvas into a numpy array](analysis/graph_to_numpy/graph_to_numpy.py)
    Convert a matplotlib.figure object into a numpy.array.

- [Set of smoothing algorithms for 1D-array](analysis/smooth_kit/smoothkit.py)
    Collection of smoothing algorithms for time series data.

### Excel

- [Convert Excel sheet to CSV files](excel/convert_to_csvs/sheet_to_csvfile.py)
    Convert each sheet of a Excel workbook to a CSV file.

- [Insert Image to Excel](excel/insert_image_to_cell/insert_image_to_cell.py)
    Insert an image into a specific cell in an Excel file using `openpyxl`.
