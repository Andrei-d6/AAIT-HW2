# AAIT-HW2

# Project dependencies
Pytorch version **1.12.1+cu113** (at least, should your for newer versions aswell).

```bash
# Example for Windows 11 install
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

Fastai library
```bash
pip install fastai
```

CleanLab framework
```bash
pip install cleanlab
```

# Task 1

# Task 2




# Project Structure
The directory structure for this project looks like this:
```
├── data                   <- Project data
├── models                 <- Directory for checkpointing
│
├── Experimental_report     <- A brief overview of the work from this project
│
├── Trained                <- Directory for training notebooks 
│   ├── v1                 <- Training results for version 1 (with annotations, configs, etc.)
│   └── v2                 <- Training results for version 2
│
├── configs                <-  Learner YAML configuration files
│
├── src                    <- Source code
│   ├── data.py            <- fastai learner DataLoaders
│   ├── generate_labels.py <- Script for creating training annotations for Task 1
│   ├── learner.py         <- Learners for both tasks (learner = model + dataloaders)
│   └── utils.py           <- Utility functions for trainig, checkpointing, saving results
│
├── submissions            <- The results for each development step
│
├── Task1.ipynb            <- Notebook with instructions for replicating results for Task 1
├── Task2.ipynb            <- Notebook with instructions for replicating results for Task 2
│
├── .gitignore             <- List of files ignored by git
└── README.md
```