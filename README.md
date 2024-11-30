# Juxtaposing Approximations Robustly and Empirically for Comparisons (JAREC)

## Overview
![overview_cleaned](https://github.com/user-attachments/assets/f3da5df0-ba52-4faa-8e50-6d3d887054f2)

## Introduction 
This repo is for a manifold approximation comparison framework called JAREC. From some projective dimensionality reduction method, you may wish to test the hypothesis $H_0: \hat{M}_1 = \hat{M}_2$, that two manifold approximations are equal to each other. In the example above, if your dataset is collected from cell cycle covariates, then you can understand the manifold as a low-dimensional representation of an entire cell cycle process. This hypothesis test could be performed between different experimental conditions, treatments, or other perturbations. JAREC does this without any distributional assumptions using bootstrapped hypothesis tests. Code to carry out this analysis is listed here in this repo.

## Installation and Example Usages
First, clone the git repository:
```
git clone https://github.com/PurvisLabTeam/JAREC.git
```

An example of how to use JAREC while tuning for biological noise present in data is in [jarec_example.ipynb](./jarec_example.ipynb). 

To simulate data to test the effectiveness of JAREC, we simulate data along a hypersphere, approximating it with spherical principal component analysis (SPCA), and systematically vary parameters of the sphere (center, radius, and subspace) to demonstrate the sensitivity of JAREC. Code to carry out the simulations is seen in [sphere_sims_jarec.py](./sphere_sims_jarec.py) with code to reproduce figures in [simulation_figures_jarec.ipynb](./simulation_figures_jarec.ipynb).

Lastly, to investigate drivers of the change from manifold found to be different with JAREC, we carry out clustering along a hypersphere using a mixture of von-Mises Fisher distributions using the moVMF library in R. Code to do this is in the [mix_VMFs_jarec.Rmd](./mix_VMFs_jarec.Rmd), with code to reproduce the figures in the [vmf_plots_jarec.ipynb](./vmf_plots_jarec.ipynb).

## Dependencies 
Data analysis was performed using Python (3.11.5) and R (4.3.0). NumPy (1.25.2), pandas (2.1.0), SciPy (1.11.2), scikit-learn (1.3.0), AnnData (0.10.3), sketchKH (0.1.1), joblib (1.3.2), and tqdm (4.66.1) Python libraries were used to perform statistical analyses. movMF (0.2-8) R packages were used to simulate von Mises-Fisher distributions. seaborn (0.13.0) and matplotlib (3.7.2) Python libraries were used for data visualization. Up to date implementations of SPCA in Python can be found at [https://github.com/purvislab/SingleCell_HyperSphere](https://github.com/purvislab/SingleCell_HyperSphere). 

## Data Access
For our example usages, we use single-cell data from ER+/HER2- breast tumor cells (T47D) treated with varying doses of palbociclib, a CDK4/6 inhibitor. T47D triplicate data under palbociclib can be found at [https://zenodo.org/records/10063003](https://zenodo.org/records/10063003), and the six-condition T47D data can found at [https://doi.org/10.5281/zenodo.13621367](https://doi.org/10.5281/zenodo.13621367). 

## License 
This software is licensed under the [MIT license](https://opensource.org/licenses/MIT).
