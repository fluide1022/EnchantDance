from .base import BASEDataModule
from .humanml.data.dataset import Music2DanceDatasetNew


class Choreospectrum3DDataModule(BASEDataModule):

    def __init__(self,
                 cfg,
                 batch_size,
                 num_workers,
                 collate_fn=None,
                 phase="train",
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers,
                         collate_fn=collate_fn)
        self.save_hyperparameters(logger=False)
        self.name = "choreospectrum3d"
        self.Dataset = Music2DanceDatasetNew
        self.cfg = cfg
        sample_overrides = {
            "split": "val",
            "tiny": True,
            "progress_bar": False
        }
        self._sample_set = self.get_sample_set(overrides=sample_overrides)
        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats

    def feats2joints(self, features):
        return features

    def joints2feats(self, features):
        return features
