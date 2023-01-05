from fastai.vision.all import *

from src.data import get_dls_task1, get_dls_task2


__all__ = ['get_learner_task1', 'get_learner_task2']



def get_learner_task1(config: dict, iteration: int = 0) -> Learner:

    dls = get_dls_task1(config, iteration=iteration)

    seed = config['SEED']
    set_seed(seed, True)
    dls.rng.seed(seed)

    if config['ARCH'] == 'resnet50':

        arch = resnet50
        opt_func = Adam if config['OPT'] == 'adam' else ranger

        learn = vision_learner(
            dls,
            arch,
            opt_func=opt_func,
            metrics=[accuracy, top_k_accuracy],
            loss_func=CrossEntropyLossFlat()
        )

    if config['FP16']:
        learn = learn.to_fp16()

    return learn


def get_learner_task2(config: dict) -> Learner:

    dls = get_dls_task2(config)

    seed = config['SEED']
    set_seed(seed, True)
    dls.rng.seed(seed)

    if config['ARCH'] == 'resnet50':

        arch = resnet50
        opt_func = Adam if config['OPT'] == 'adam' else ranger

        learn = vision_learner(
            dls,
            arch,
            opt_func=opt_func,
            metrics=[accuracy, top_k_accuracy],
            loss_func=LabelSmoothingCrossEntropy()
        )
    else:

        n_out, sa, sym = config['N_OUT'], config['SA'], config['SYM']
        pool = MaxPool if config['POOL'] == 'max_pool' else AvgPool
        act_fn = defaults.activation if config['ACT_FN'] == 'relu' else Mish

        net = xresnext50(pretrained=True, n_out=n_out, sa=sa,
                        sym=sym, pool=pool, act_cls=act_fn)

        opt_func = None
        mom, sqr_mom, eps = config['MOM'], config['SQR_MOM'], config['EPS']

        if config['OPT'] == 'adam':
            opt_func = partial(Adam, mom=mom, sqr_mom=sqr_mom, eps=eps)
        else:
            opt_func = partial(ranger, mom=mom, sqr_mom=sqr_mom, eps=eps, beta=config['BETA'])

        learn = Learner(
            dls,
            net,
            opt_func=opt_func,
            metrics=[accuracy, top_k_accuracy],
            loss_func=LabelSmoothingCrossEntropy()
        )

    if config['FP16']:
        learn = learn.to_fp16()

    return learn