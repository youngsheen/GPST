# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import logging
import torch

logger = logging.getLogger(__name__)


class ManifestDataset(torch.utils.data.Dataset):
    def __init__(self, manifest):
        with open(manifest, "r") as fin:
            self.files = [x.strip() for x in fin.readlines()]

        logger.info(
            f"containing {len(self.files)} files"
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return pathlib.Path(self.files[index])
