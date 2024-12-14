# risk-taking_behavior_2024

This repository contains the code and scripts used to generate the spatial mapping results reported in the study *"Neural signatures of opioid-induced risk-taking behavior in the prelimbic prefrontal cortex"* by Cana B. Quave, Andres M. Vasquez, Guillermo Aquino-Miranda, Milagros Marín, Esha P. Bora, Chinenye L. Chidomere, Xu O. Zhang, Douglas S. Engelke, Fabricio H. Do-Monte (2024). DOI: [10.1016/j.biopsych.2024.01.020](https://doi.org/10.1016/j.biopsych.2024.01.020). The repository also includes detailed outputs from the spatial mapping analyses presented in the paper.

# Content

- **/paper**: Original scientific article.
- **/src**: Code for running the spatial mapping analysis.
- **/log_output**: Outputs from the analysis.

# Workflow Description

The main analysis script is located in `src/spatial_mapping.py`. This script requires electrophysiological (ephys) data and body positions tracked with DeepLabCut from videos recorded for each animal and task. Then, it will analyze the animal's neuronal activity and spatial location of the animal. 

![Screenshot 2024-11-18 at 13 14 16](https://github.com/user-attachments/assets/3a5a5c33-8caa-4d02-9d58-327fbb7727f3)

Spatial and temporal alignment to frames for neuronal and behavioral data: 
![Screenshot 2024-11-18 at 13 55 26](https://github.com/user-attachments/assets/9bf88f9d-29cc-4d88-9461-ad972e3a7313)


### Main steps of the script
1. Import necessary libraries.
2. Define the functions to run the pipeline.
3. Set analysis variables based on task type (Conflict or Preference), including likelihood thresholds, number of bins, input files, etc.
4. Load inputs: ephys files, video files (`.mp4`), and body tracking data (`.csv`), along with metadata such as frames per second (fps), delay, and paired side.
5. Run analyses for each animal.
6. Generate outputs:
   - `.pkl` files with z-score matrices for excitatory and inhibitory responses by group (risk-takers, risk-avoiders, saline group).
   - A log file (`.txt`) detailing analysis steps and results.
   - Figures showing body tracking and likelihood from DeepLabCut, spatial mapping, and time and z-score distributions per bin.

# Citation

If you use this code, please cite the associated paper:

```
Cana B. Quave, Andres M. Vasquez, Guillermo Aquino-Miranda1, Milagros Marín, Esha P. Bora, Chinenye L. Chidomere, Xu O. Zhang, Douglas S. Engelke, Fabricio H. Do-Monte (2024), Neural signatures of opioid-induced risk-taking behavior in the prelimbic prefrontal cortex, bioRxiv (https://doi.org/10.1101/2024.02.05.578828)
```
