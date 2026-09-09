# STMHybridModel
A hybrid model describing the biological dynamics of STM in pluripotent/differentiated cells.


## Introduction
The python scripts (interface: `main.py`) provid a CLI style program, for simulation and visualization of a hybrid model constructed at *SHOOT MERISTEMLESS* (*STM*) gene locus in *Arabidopsis thaliana*. The model combines multiple bilogical process, e.g., gene expression, chromatin modification (mainly methylation), protein degration and cell division, etc., in a whole system to explain some experimental observations. To simulate the series of biological events, we use the **Gillespie's stochastic simulation algorithm** (**SSA**).  

This model is adopted in the following paper:
- Cell cycle-driven epigenetic resetting maintains meristematic cell fate for shoot branching. (**under review**)

An example of implementing SSA in biocircuits is presented in **[biocircuits_ssa_tutorial.ipynb](./assets/biocircuits_ssa_tutorial.ipynb)**. For a elaborate introduction of this model, see **[STMHybridModel_intro.pdf](./assets/STMHybridModel_intro-latest.pdf)**.


This program contains 6 subcommands to perform different tasks:

- ***schmdg*** : Plot schematic diagram to explain model mathematical principles
- ***epistb*** : Plot the profiles of methylation level and *STM* expression change over multiple days in both stem cells and differentiated cells to show epigenetic stability
- ***dynmcyc*** : Plot the influence of dynamic (increasing/decreasing) cell cycle on stem cells
- ***divarrest*** : Plot transition curves of stem cell differentiation during different length of division arrest & cell type distribution statistics after division recovery
- ***rescue*** : Plot recovery curves of differentiated cells after divison arrest using different rescue stategies
- ***bimap***: Plot bistability heatmap in 2-D parameter $k_{\text{me}}$ - $P_{\text{dem}}$ space under different cell cycle conditions

Before running, make sure that your environment contains the necessary dependencies, which are listed in the `pyproject.toml` file.


## Example Usage
Here we provide some examples of running the programs.

You can use `-h/--help` option to get help information about the subcommands and their argument hints:
```shell
python main.py -h

python main.py schmdg -h
python main.py divarrest -h
```

### schmdg
```shell
# schematic diagram

# Extended Data Fig. 9b and 11
python main.py schmdg -d  
# -d/--display: display all plotted figures before the program exits
```

### epistb
```shell
# epigenetic stability

# Extended Data Fig. 9f-g
python main.py epistb -ns 80 -et 0 3 -md 40 -c -d
# -ns/--num_samples: number of model samples
# -et/--epi_tag: initial methylation state
# -md/--monitor_days: days of monitoring in simulation
# -c/--concise: concise mode with additional analysis disabled

# With additional plot -> evolution heatmap
python main.py epistb -ns 80 -et 0 3 -md 40 -ts 10 -d -t
# -ts/--timeid_step: step of time index in evolution heatmap plot
# -t/--temp: export files to command temp folder
```

### dynmcyc
```shell
# dynamic cell cycle

# Extended Data Fig. 9h-i
python main.py dynmcyc -ns 80 -df 0.5 3 -ec 12 -md 30 -d
# -df/--dynamic_factor: dynamic factor
# -ec/--equi_cycles: cell cycle numbers in pre-equilibrium
```

### divarrest
```shell
# division arrest

# Fig. 4b-c and Extended Data Fig. 10a-h
python main.py divarrest -ns 80 -ec 12 -md 35 -ad 1 3 5 7 9 11 13 15 17 19 -c -d
# -ad/--arrest_days: days of cell division arrest

# With additional plot -> cell type distribution curves after division restart
python main.py divarrest -ns 800 -ec 12 -md 35 -ad 1 3 5 7 9 11 13 15 17 19 -ps 10 -cd 8 16 -d -t
# -ps/--plot_step: index step of sample that to be show in plot
# -cd/--count_days: days to count cell type distribution after division restart
```
**Note:** If you want to explore the **alternative model** (Extended Data Fig. 12a-d), in which the cycle-related parameters are excluded, please turn to function `_calc_beta` (**line 81-108**) in `util_funcs.py`, and **swap the comment state between line 105 and 106** before running the scripts.

### rescue
```shell
# rescue experiments

# Fig. 5a-b
python main.py rescue -ns 80 -rs M A -pr 0.85 -ag 0.05 -ad 20 -td 25 -rd 5 -d
# -rs/--rescue_strategy: rescue strategy: M/m - remove methylation, A/a - add ATH1, S/s - add STM
# -pr/--prob_removal: probability of methylation removal
# -ag/--affn_growth: growth ratio of transcrition factor affinity to gene locus
# -td/--treat_days: days of rescue treatment
# -rd/--rest_days: days of resting state before rescue
```

### bimap
```shell
# bistability heatmap

# Extended Data Fig. 9c-e
python main.py bimap -ns 100 -cc 11.0 22.0 66.0 -nc 40 -mp 101 -d
# -cc/--cell_cycle: cell cycle
# -nc/--num_cycles: number of cell cycles in simulation
# -mp/--max_pixel: pixels in both dimension
```
