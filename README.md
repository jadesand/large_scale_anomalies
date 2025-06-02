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
@ARTICLE{Shi2023,
       author = {{Shi}, Rui and {Marriage}, Tobias A. and {Appel}, John W. and {Bennett}, Charles L. and {Chuss}, David T. and {Cleary}, Joseph and {Eimer}, Joseph R. and {Dahal}, Sumit and {Datta}, Rahul and {Espinoza}, Francisco and {Li}, Yunyang and {Miller}, Nathan J. and {N{\'u}{\~n}ez}, Carolina and {Padilla}, Ivan L. and {Petroff}, Matthew A. and {Valle}, Deniz A.~N. and {Wollack}, Edward J. and {Xu}, Zhilei},
        title = "{Testing Cosmic Microwave Background Anomalies in E-mode Polarization with Current and Future Data}",
      journal = {\apj},
     keywords = {Cosmic microwave background radiation, Early universe, Observational cosmology, 322, 435, 1146, Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2023,
        month = mar,
       volume = {945},
       number = {1},
          eid = {79},
        pages = {79},
          doi = {10.3847/1538-4357/acb339},
archivePrefix = {arXiv},
       eprint = {2206.05920},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023ApJ...945...79S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
