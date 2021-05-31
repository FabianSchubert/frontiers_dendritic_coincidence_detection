# nonlinear_dendritic_coincidence_detection

This repository contains the simulation and visualization code for a two-compartment neuron model
equipped with Hebbian plasticity rules.

## Instructions for creating the Figures

The following commands should be run from the base folder of the repository.

* Figure 1: 
```sh
python3 -m plotting.comp_model
```
* Figure 3:
```sh
python3 -m simulation.correlation_dimension_scaling
python3 -m simulation.correlation_dimension_scaling_bcm
python3 -m plotting.correlation_dimension_scaling_composite
```
* Figure 4:
```sh
python3 -m simulation.classification_dimension_scaling
python3 -m simulation.classification_dimension_scaling_bcm
python3 -m plotting.classification_dimension_scaling_composite
```
* Figure 5:
```sh
python3 -m simulation.classification_correlation_dimension_scaling
python3 -m simulation.classification_correlation_dimension_scaling_bcm
python3 -m plotting.classification_correlation_dimension_scaling_composite
```
* Figure 6:
```sh
python3 -m plotting.obj_func_illustration
```

When one of the scripts in the simulation or plotting module is run for the first time, it will create the respective /data and /plots folders in the base directory.
