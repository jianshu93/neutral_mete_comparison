METE-SSNT
===========

Research project that compares models of the Maximum Entropy Theory of Ecology (METE) and of the size-structured neutral theory (SSNT), using forest survey data.
The project has been developed by Xiao Xiao, James P. O'Dwyer, and Ethan P. White.
Our manuscript has been accepted by Ecology. 
A preprint is available on arXiv (http://arxiv.org/abs/1502.02767). 

Code in this repository replicates all analyses in the manuscript. 
A subset of the data sets included in our study can be obtained from our previous Dryad submission (http://datadryad.org/handle/10255/dryad.71012).

Setup
------------
Requirements: Python 2.x, and the following Python modules: `numpy`, `scipy`, `matplotlib`, `mpl_tookits`, `mpmath`, `multiprocessing`, and `Cython`. 
In addition, the following custom Python modules are also required: `METE` (https://github.com/weecology/METE),  and `macroecotools` (https://github.com/weecology/macroecotools).
Note that you will have to navigate to /mete_distributions under module `METE` and run `setup.py` (from the command line: `python setup.py`) for Cython to properly compile.

Replicate analyses
------------------
Obtain sample data sets from http://datadryad.org/handle/10255/dryad.71012, and save them under the subdirectory /data/ in the working directory. 

Save the two scripts, `ssnt_mete_comparison.py` and `ssnt_mete_comp_analysis.py` under the working directory.

All analyses can be replicated by running the following command from the command line: 

`python ssnt_mete_comp_analysis.py`

By default, figures will be saved to the subdirectory /out\_figs/. 
Intermediate output files will be saved to the subdirectory /out\_files/. 
