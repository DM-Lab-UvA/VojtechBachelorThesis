# Vojta's Bachelor project

This repository contains the code that I have used for data analysis in my Bachelor thesis at Amsterdam University College. The code uses Python 3 and
requires the following packages to be installed:

+ `h5py`
+ `matplotlib`
+ `numpy`
+ `pandas`
+ `pybind11`
+ `scikit-learn`
+ `scipy`

Additionally, in my analysis, I have used a custom Python package for a greedy search for the best Minimally Complex Model (MCM), 
available at [1](https://github.com/DM-Lab-UvA/MinCompSpin_Python/tree/Reorganisation). Note that I have used the
"Reorganisation" branch of the MinCompSpin_Python repository, and I my code assumes that the folder with this repository is renamed
to "MinCompSpin_Python" and placed in the same folder as the rest of my code when it is run. In my project, I have analyzed data 
obtained by [2](https://www.nature.com/articles/s41467-022-30600-4). All of their data is available at 
[3](https://gitlab.com/csnlab/olcese-lab/modid-project/2nd-bump/-/commits/master/?ref_type=HEADS). 

## Works cited
1. de Mulatier C. Greedy algorithm for detecting community structure in binary data [Internet]. GitHub. 
2024 [cited 2024 Mar 20]. Available from: [https://github.com/clelidm/MinCompSpin_Greedy](https://github.com/clelidm/MinCompSpin_Greedy)<br/>
2. Oude Lohuis MN, Pie JL, Marchesi P, Montijn JS, de Kock CPJ, Pennartz CMA, et al. Multisensory task 
demands temporally extend the causal requirement for visual cortex in perception. Nature Communications. 2022 May 23;13(1). <br/>
3. Olcese U, Jean. csnlab / olcese-lab / modid-project / 2nd-bump · GitLab [Internet]. GitLab. 2022 [cited 2024 May 27]. 
Available from: [https://gitlab.com/csnlab/olcese-lab/modid-project/2nd-bump](https://gitlab.com/csnlab/olcese-lab/modid-project/2nd-bump)
