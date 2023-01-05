from fastai.vision.all import *


__all__ = ['get_dls_task1', 'get_dls_task2']


def get_dls_task1(config: dict, iteration: int = 0) -> DataLoaders:

    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    col_x, col_y, col_valid = config['COL_IMAGE_PATH'], config['COL_LABEL'], config['COL_VALID']

    size, min_scale, flip_item_p = config['SIZE'], config['MIN_SCALE'], config['FLIP_ITEM']
    bs, num_workers = config['BATCH_SIZE'], config['NUM_WORKERS']
    pref = config['PATH_BASE_DATA']
    seed = config['SEED']

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
	                   splitter=ColSplitter(col=col_valid),
	                   get_x=ColReader(col_x, pref=pref),
	                   get_y=ColReader(col_y),
	                   item_tfms=[RandomResizedCrop(size, min_scale=min_scale),
	                              FlipItem(flip_item_p)],
	                   batch_tfms=batch_tfms)

    path_labeled_annotations = f"{config['PATH_BASE_DATA']}/{config['PATH_LABELED_ANNOTATIONS']}"
    path_unlabeled_annotations = f"{config['PATH_BASE_DATA']}/{config['PATH_UNLABLED_ANNOTATIONS']}"

    if iteration >= 2:
        split = path_unlabeled_annotations.split('.')
        path_unlabeled_annotations = f"{split[0]}_{iteration-1}.{split[1]}"

    df = pd.read_csv(path_labeled_annotations)

    if iteration >= 1 and os.path.exists(path_unlabeled_annotations):
        df_unlabeled = pd.read_csv(path_unlabeled_annotations)
        df = pd.concat([df, df_unlabeled])

    return dblock.dataloaders(df, bs=bs, num_workers=num_workers)



def get_dls_task2(config: dict) -> DataLoaders:

    batch_tfms = [Normalize.from_stats(*imagenet_stats)]

    if config['RANDOM_ERASE_P']:
        p, max_count = config['RANDOM_ERASE_P'], config['MAX_COUNT']
        batch_tfms.append(RandomErasing(p=p, max_count=max_count))

    col_x, col_y = config['COL_IMAGE_PATH'], config['COL_LABEL']

    size, min_scale, flip_item_p = config['SIZE'], config['MIN_SCALE'], config['FLIP_ITEM']
    bs, num_workers = config['BATCH_SIZE'], config['NUM_WORKERS']
    valid_pct, seed = config['VALID_PCT'], config['SEED']
    pref = config['PATH_BASE_DATA']

    dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       splitter=RandomSplitter(valid_pct=valid_pct, seed=seed),
                       get_x=ColReader(col_x, pref=pref),
                       get_y=ColReader(col_y),
                       item_tfms=[RandomResizedCrop(size, min_scale=min_scale),
                                  FlipItem(flip_item_p)],
                       batch_tfms=batch_tfms)

    path_annotations = f"{config['PATH_BASE_DATA']}/{config['PATH_LABELED_ANNOTATIONS']}"
    df = pd.read_csv(path_annotations)

    return dblock.dataloaders(df, bs=bs, num_workers=num_workers)
