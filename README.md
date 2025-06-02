# large_scale_anomalies
<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
A Python package for computing various large-scale anomaly estimators in Cosmic Microwave Background (CMB) data.

## Overview
This package implements various estimators to quantify large-scale anomalies in the Cosmic Microwave Background (CMB) as described in Shi et al. 2023. It includes tools for analyzing:
- Lack of correlation
- Quadrupole-Octupole alignment
- Odd-parity asymmetry
- Hemispherical power asymmetry

## Installation
```
git clone https://github.com/username/large_scale_anomalies.git
cd large_scale_anomalies
pip install -e .
```

## Usage
See `example.ipynb`

## Dependencies
- numpy
- scipy
- healpy

## Citation
If you use this code in your research, please cite:
```
@article{Shi2023,
  author = {Shi, U R and others},
  title = {CMB Large Scale Anomalies},
  journal = {The Astrophysical Journal},
  volume = {},
  year = {2023},
  doi = {10.3847/1538-4357/acb339}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
