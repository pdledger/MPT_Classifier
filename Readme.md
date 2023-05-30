# MPT_Classifier

A classifier based on MPT spectral signatures.


## Motivation

The motivation of this repository is to host a working version of the MPT classifier written by Ben Wilson and to eventually move towards more complex. elaborate classification work. I.e. testing on real measurement data.


## Repository overview

The repository is structured as follows:
```bash
├── README.md
├── Trainer.py
├── Creator.py
├── External_File_Loader.py
├── Classes
│   ├── Class_Realistic
│       ├── Class_Clothing
│       │    ├── Class_Belt_Buckle
│       │    │     ├── OBJ_Flat_Buckle
│       │    │     │     ├── al_0.001_mu_5_sig_1.6e6
│       │    │     │     │     ├──1-1e10_81_el_28837_ord_4_POD_13_1e-4
│       │    │     │     │     │     ├──Data
│       │    │     │     │     │     ├──Functions
│       │    │     │     │     │     ├──Graphs
│       │    │     │     │     │     ├──Input_files
│       │    │     │     │     │     ├──PlotEditior.py
│       │    │     │     │     │     ├──PlotterSettings.py
│       │    │     │     │     │     └──PODPlotEditor.py
│       │    │     │     │     └── ...
│       │    │     │     └── ...
│       │    │     └── ...
│       │    └── ...
│       ├── Class_Coins
│       ├── Class_Coins
│       └── ...
│
└── src
    ├── analysis
    ├── data-preparation
    └── paper

```
## Running instructions

Explain to potential users how to run/replicate your workflow. If necessary, touch upon the required input data, which secret credentials are required (and how to obtain them), which software tools are needed to run the workflow (including links to the installation instructions), and how to run the workflow.


## More resources

Point interested users to any related literature and/or documentation.


## About

Explain who has contributed to the repository. You can say it has been part of a class you've taken at Tilburg University.

