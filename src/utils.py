from fastai.callback.tracker import SaveModelCallback
from fastai.data.transforms import get_image_files
from typing import Optional, List, Callable
from fastai.torch_core import set_seed
from fastai.learner import Learner
from pathlib import Path


import pandas as pd
import torch
import yaml


__all__ = ['seed_everything', 'load_configuration', 'do_fit', 'save_preds']



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
        torch.save(learn.model.state_dict(), f"{learn.model_dir}/{save_name}.pth")

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