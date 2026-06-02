# STMHybridModel
A hybrid model describing the biological dynamics of STM in pluripotent/differentiated cells.


## Introduction
The python scripts (interface: `main.py`) provid a CLI style program, for simulation and visualization of a hybrid model constructed at *SHOOT MERISTEMLESS* (*STM*) gene locus in *Arabidopsis thaliana*. The model combines multiple bilogical process, e.g., gene expression, chromatin modification (mainly methylation), protein degration and cell division, etc., in a whole system to explain some experimental observations. To simulate the series of biological events, we use the **Gillespie's stochastic simulation algorithm** (**SSA**).  
 
This model is adopted in the following paper:
- Guo Y., Yang Y. Shi B. et al. A cell cycle-dependent epigenetic Sisyphus mechanism maintains stem cell fate for shoot branching. (**under review**)

An example of implementing SSA in biocircuits is presented in **[biocircuits_ssa_tutorial.ipynb](./biocircuits_ssa_tutorial.ipynb)**. For a elaborate introduction of this model, see **[STMHybridModel_intro.pdf](./assets/STMHybridModel_intro.pdf)**.


## Usage
This program contains 6 subcommands to prtform different tasks:

- ***schmdg*** : plot schematic diagram to explain model mathematical principles
- ***epistb*** : plot the profiles of methylation level and *STM* expression change over multiple days in both stem cells and differentiated cells to show epigenetic stability
- ***dynmcyc*** : plot the influence of dynamic (increasing/decreasing) cell cycle on stem cells
- ***divarrest*** : plot transition curves of stem cell differentiation during different length of division arrest & cell type distribution statistics after division recovery
- ***rescue*** : plot recovery curves of differentiated cells after divison arrest using different rescue stategies
- ***bimap***: plot bistability heatmap in 2-D parameter $k_{me}$ - $P_{dem}$ space under different cell cycle conditions

Using `-h/--help` option to get helps about the script usage and argument hints. Before using, make sure that your environment contains the necessary dependencies.

**Note:** If you want to explore the **alternative model**, in which the cycle-related parameters are excluded, please find the following snippet (**line 105-106**) in `util_funcs.py`

    beta = beta_max * (1 - math.exp(-b * T / T_0))
    # beta = beta_max * (1 - math.exp(-b))  # without cell cycle-dependency

and **swap the comment state between line 105 and 106** before running the scripts.


### Examples
    python main.py -h
    python main.py schmdg -h
    python main.py divarrest -h
    
    python main.py schmdg -d
    python main.py epistb -ns 80 -et 0 3 -md 40 -ts 10 -d
    python main.py epistb -ns 80 -et 0 3 -md 40 -c -d -t
    python main.py dynmcyc -ns 80 -df 0.5 3 -ec 12 -md 30 -d
    python main.py divarrest -ns 800 -ec 12 -md 35 -ad 1 3 5 7 9 11 13 15 17 19 -ps 10 -cd 8 16 -d
    python main.py divarrest -ns 80 -ec 12 -md 35 -ad 1 3 5 7 9 11 13 15 17 19 -a -c -d -t
    python main.py rescue -ns 80 -rs M A S -ad 20 -td 25 -rd 5 -d
    python main.py bimap -ns 100 -cc 11.0 22.0 44.0 66.0 -nc 40 -mp 101 -d
