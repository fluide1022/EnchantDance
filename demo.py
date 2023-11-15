import os
from pathlib import Path
import numpy as np
import torch

from enchant.config import parse_args
from enchant.data.get_data import get_datasets
from enchant.models.get_model import get_model
from enchant.utils.logger import create_logger


def main(args):
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")
    task =args.task

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "samples_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create model
    model = get_model(cfg, dataset)

    # loading checkpoints
    state_dict = torch.load(cfg.TEST.CHECKPOINTS, map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    with torch.no_grad():
        for rep in range(cfg.NUMS):
            for id, batch in enumerate(dataset.test_dataloader()):
                if task == "reconstrucion":
                    batch["motion"] = batch["motion"].to(device)
                    length = batch["length"]
                    joints, joints_ref = model.recon_from_motion(batch)
                elif task == "music_dance":
                    batch["motion"] = batch["motion"].to(device)
                    joints = model(batch)

                nsample = len(joints)
                length = batch["length"]
                for i in range(nsample):
                    npypath = str(output_dir / f"{task}_{length[i]}_batch{id}_{i}_{rep}.npy")
                    np.save(npypath, joints[i].detach().cpu().numpy())


if __name__ == "__main__":
    # add argument
    import argparse
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--task', default="music_dance", help='sum the integers')
    args = parser.parse_args()
    main(args)
