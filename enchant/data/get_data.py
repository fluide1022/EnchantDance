from os.path import join as pjoin
from .Choreospectrum3D import Choreospectrum3DDataModule
from .utils import *

def get_collate_fn(name, phase="train"):
    return enchant_collate


# map config name to module&path
dataset_module_map = {
    "choreospectrum3d": Choreospectrum3DDataModule,
}

def get_datasets(cfg, logger=None, phase="train"):
    # get dataset names form cfg
    dataset_names = eval(f"cfg.{phase.upper()}.DATASETS")
    datasets = []
    for dataset_name in dataset_names:
        if dataset_name.lower() in ["choreospectrum3d", "aist"]:
            data_root = eval(f"cfg.DATASET.{dataset_name.upper()}.ROOT")
            # get collect_fn
            collate_fn = get_collate_fn(dataset_name, phase)
            # get dataset module
            dataset = dataset_module_map[dataset_name.lower()](
                cfg=cfg,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                num_workers=cfg.TRAIN.NUM_WORKERS,
                debug=cfg.DEBUG,
                collate_fn=collate_fn,
                music_dir=pjoin(data_root, "feature"),
                motion_dir=pjoin(data_root, 'smpl'),
                max_motion_length=cfg.DATASET.SAMPLER.MAX_LEN,
                min_motion_length=cfg.DATASET.SAMPLER.MIN_LEN,
            )
            datasets.append(dataset)
        else:
            raise NotImplementedError
    cfg.DATASET.NFEATS = datasets[0].nfeats
    return datasets
