from fastai.callback.tracker import SaveModelCallback
from fastai.data.transforms import get_image_files
from typing import Optional, List, Callable
from torchvision.models import ResNet50_Weights
from fastai.learner import load_learner
from fastai.torch_core import set_seed
from fastai.learner import Learner

from cleanlab.filter import find_label_issues
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import yaml
import os


__all__ = ['seed_everything', 'load_configuration', 'do_fit', 
           'save_preds', 'save_clean_labels', 'create_submission']



def seed_everything(dls, seed: int = 42):
    set_seed(seed, True)
    dls.rng.seed(seed)


def load_configuration(config_file_path: str) -> dict:
    with open(config_file_path) as config_file:
        return yaml.safe_load(config_file)


def do_fit(
    learn: Learner,
    save_name: str,
    fname: str = 'autosaved',
    fit_type: str = 'flat_cos', # 'one_cycle'
    epochs: int = 3,
    lr: float = 1e-3,
    pct_start: float = 0.9, # should be under 1.0
    wd: float = 1e-3,
    cbs: Optional[List[Callable]] = None,
    save_state_dict: bool = False,
    show_results: bool = False
):


    cbs = [] if cbs is None else cbs
    cbs = cbs + [SaveModelCallback(fname=fname)]

    if fit_type == 'one_cycle':
        learn.fit_one_cycle(
            epochs,
            lr_max=lr,
            pct_start=pct_start,
            cbs=cbs,
            wd=wd)
    else:
        learn.fit_flat_cos(
            epochs,
            lr=lr,
            pct_start=pct_start,
            cbs=cbs,
            wd=wd)

    learn.save(save_name)

    if save_state_dict:
        learn.export(f"{learn.model_dir}/{save_name}.pkl")
        torch.save(learn.model.state_dict(), f"{learn.model_dir}/{save_name}_dict.pth")

    if show_results:
        learn.show_results(max_n=1, figsize=(5, 5))


def save_preds(learn: Learner, config: dict, iteration: int = None):

    # Read in the unlabeled images
    unlabeled_images = get_image_files(config['PATH_UNLABELED_IMAGES'])

    # Create the dataloader
    dl = learn.dls.test_dl(unlabeled_images)

    # Get the predictions
    preds, _ = learn.get_preds(dl=dl)
    preds = preds.argmax(dim=1)

    # Construct relative paths
    paths = [Path(*p.parts[1:]) for p in unlabeled_images]

    # Create unlabeled DataFrame
    col_x, col_y, col_valid = config['COL_IMAGE_PATH'], config['COL_LABEL'], config['COL_VALID']

    df = pd.DataFrame(list(zip(paths, preds.numpy())), columns=[col_x, col_y])
    df[col_valid] = False

    save_path = f"{config['PATH_BASE_DATA']}/{config['PATH_UNLABLED_ANNOTATIONS']}"

    if iteration is not None:
        split = save_path.split('.')
        save_path = f"{split[0]}_{iteration}.{split[1]}"

    df.to_csv(save_path, index=False)


def save_clean_labels(learn: Learner, config: dict):
    
    path_annotations = f"{config['PATH_BASE_DATA']}/{config['PATH_LABELED_ANNOTATIONS']}"
    df = pd.read_csv(path_annotations)
    
    labels = df['label_idx'].values
    
    path_images = f"{config['PATH_BASE_DATA']}/{config['PATH_LABELED_IMAGES']}"
    unlabeled_images = get_image_files(path_images)
    
    unlabeled_images = sorted(unlabeled_images, key=lambda x: len(x.name))
    
    dl = learn.dls.test_dl(unlabeled_images)
    preds, _ = learn.get_preds(dl=dl)
    
    labels = np.array(labels)
    preds  = np.array(preds)
    
    ordered_label_issues = find_label_issues(
        labels=labels,
        pred_probs=preds,
        return_indices_ranked_by='self_confidence',
    )
    
    error_df = df.loc[ordered_label_issues]
    clean_idx = [idx for idx, _ in enumerate(df.values) if not idx in ordered_label_issues]
    
    clean_df = df.loc[clean_idx]
    
    save_path = f"{config['PATH_BASE_DATA']}/{config['PATH_CLEAN_ANNOTATIONS']}"
    clean_df.to_csv(save_path, index=False)


def create_submission(
    path_learn: str, 
    path_test_images: str,
    submission_name: str,
    base_model_dir: str = 'models',
    base_save_dir: str = 'submissions',
    cpu: bool = False
):
    
    learn = load_learner(f"{base_model_dir}/{path_learn}", cpu=cpu)
    test_images = get_image_files(path_test_images)
    test_images = sorted(test_images, key=lambda x: len(x.name))

    test_dl = learn.dls.test_dl(test_images)
    preds, _ = learn.get_preds(dl=test_dl)
    preds = preds.argmax(dim=1)

    paths = [path.name for path in test_images]
    df = pd.DataFrame(list(zip(paths, preds.numpy())), columns=['sample', 'label'])
    df.to_csv(f"{base_save_dir}/{submission_name}", index=False)