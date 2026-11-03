import logging
import os

import pandas as pd
import torch

file_dir = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)


def load_gps_data(csv_file):
    """Load a ``LAT,LON`` CSV into a float32 tensor of shape (n, 2)."""
    data = pd.read_csv(csv_file)
    log.info("gps_gallery: %s (%d punti)", csv_file, len(data))
    return torch.tensor(data[["LAT", "LON"]].values, dtype=torch.float32)
