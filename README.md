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

# Tasks
The tasks covered in this project involve the problem of image classification when encountering datasets that are not as neat and tidy - some of the samples are not annotated or the target labels are just wrong.<br>
For a detailed explanation of how each task is addressed please refer to the [**Experimental report**](Experimental_report/Experimental_report_Andrei_Dugaesescu_IA2.pdf).

## Task 1
Image classification with missing labels. The core idea is to pseudo-label the samples with missing labels and then retrain the network on the new dataset. <br>
For reproducing the results obtained for this task please follow the steps layed out in [**Task1.ipynb**](Task1.ipynb). The entire training process results (including intermediary steps) can be found in [**Trained/v2/Task1.ipynb**](Trained/v2/Task1.ipynb).

### ! Important
Before running the training steps presented in [**Task1.ipynb**](Task1.ipynb) please make sure to run the [**generate_labels.py**](src/generate_labels.py) in order to create the validation set from the labeled samples for Task 1. A quick rundown of how that script is intended to be used can be seen below.

```
python generate_labels.py -h
usage: generate_labels.py [-h] [--seed SEED] [--valid_pct VALID_PCT] --in_file IN_FILE --out_file OUT_FILE

Parser for creating the initial dataset for Task 1

options:
  -h, --help            show this help message and exit
  --seed SEED           Seed for reproducibility
  --valid_pct VALID_PCT
                        Percentage of images for the validation set from the labeled samples
  --in_file IN_FILE     Path to the original annotations
  --out_file OUT_FILE   Path to the new annotations with train/validation split

Example:
python generate_labels.py --in_file=data/task1/train_data/annotations.csv
--out_file=data/task1/train_data/annotations_labeled.csv
```

## Task 2
Image classification with noisy labels. The main idea for this task is to train a robust enough classifier that can be used for cleaning the noisy labels. Another model is then trained on the clean dataset and used for the final predictions. In order to reproduce the results obtained for this task please follow the steps layed out in [**Task2.ipynb**](Task2.ipynb). The entire training process results (including intermediary steps) can be found in [**Trained/v2/Task2.ipynb**](Trained/v2/Task2.ipynb).

<br>

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
│   └── utils.py           <- Utility functions for training, checkpointing, saving results
│
├── submissions            <- The results for each development step
│
├── Task1.ipynb            <- Notebook with instructions for replicating results for Task 1
├── Task2.ipynb            <- Notebook with instructions for replicating results for Task 2
│
├── .gitignore             <- List of files ignored by git
└── README.md
```