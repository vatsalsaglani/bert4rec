from train_pipeline import trainer
from constants import TRAIN_CONSTANTS

from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)

training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Loss", justify="center"),
    Column("Accuracy", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

valid_loggger = Table(
    Column("Epoch", justify="center"),
    Column("Loss", justify="center"),
    Column("Accuracy", justify="center"),
    title="Validation Status",
    pad_edge=False,
    box=box.ASCII,
)

loggers = dict(CONSOLE=console,
               TRAIN_LOGGER=training_logger,
               VALID_LOGGER=valid_loggger)

model_params = dict(
    SEED=3007,
    VOCAB_SIZE=9726,
    heads=4,
    layers=6,
    emb_dim=512,
    pad_id=TRAIN_CONSTANTS.PAD,
    history=TRAIN_CONSTANTS.HISTORY,
    trained=
    "../models/bert4rec-itr-1/model_files_initial/bert4rec-state-dict.pth",
    # trained=None,
    LEARNING_RATE=0,
    EPOCHS=100,
    SAVE_NAME="bert4rec.pt",
    SAVE_STATE_DICT_NAME="bert4rec-state-dict.pth",
)

data_params = dict(path="../data/ratings_mapped.csv",
                   group_by_col="userId",
                   data_col="movieId_mapped",
                   train_history=TRAIN_CONSTANTS.HISTORY,
                   valid_history=5,
                   padding_mode="right",
                   MASK=TRAIN_CONSTANTS.MASK,
                   LOADERS=dict(TRAIN=dict(batch_size=8,
                                           shuffle=True,
                                           num_workers=0),
                                VALID=dict(batch_size=4,
                                           shuffle=False,
                                           num_workers=0)))

output_dir = "../models/bert4rec-itr-2"

trainer(data_params, model_params, loggers, full_train=True)
