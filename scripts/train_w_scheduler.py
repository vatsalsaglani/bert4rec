from scheduling_pipeline import scheduler
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
    VOCAB_SIZE=59049,
    heads=4,
    layers=6,
    emb_dim=256,
    pad_id=TRAIN_CONSTANTS.PAD,
    history=TRAIN_CONSTANTS.HISTORY,
    trained=
    "/content/drive/MyDrive/bert4rec/models/rec-transformer-model-9/model_files/bert4rec-state-dict.pth",
    # trained=None,
    LEARNING_RATE=0.1,
    EPOCHS=5000,
    SAVE_NAME="bert4rec.pt",
    SAVE_STATE_DICT_NAME="bert4rec-state-dict.pth",
    CLIP=2

    # NEW_VOCAB_SIZE=59049
)

data_params = dict(
    # path="/content/bert4rec/data/ratings_mapped.csv",
    #  path="drive/MyDrive/bert4rec/data/ml-25m/ratings_mapped.csv",
    path="/content/drive/MyDrive/bert4rec/data/ml-25m/ratings_mapped.csv",
    group_by_col="userId",
    data_col="movieId_mapped",
    train_history=TRAIN_CONSTANTS.HISTORY,
    valid_history=5,
    padding_mode="right",
    MASK=TRAIN_CONSTANTS.MASK,
    chunkify=10,
    LOADERS=dict(TRAIN=dict(batch_size=64, shuffle=False, num_workers=0),
                 VALID=dict(batch_size=32, shuffle=False, num_workers=0)))

optimizer_params = {
    "OPTIM_NAME": "SGD",
    "PARAMS": {
        "lr": 0.142,
        "momentum": 0.85,
    }
}

# with ReduceLROnPlateau
sch_parameters = {
    "SCHEDULER_NAME": "ReduceLROnPlateau",
    "TYPE": "reduce",
    "DECAY_PARAM": "loss",
    "PARAMS": {
        "mode": "min",
        "factor": 0.092,
        "patience": 10,
        "verbose": True
    }
}

# sch_parameters = {
#     "SCHEDULER_NAME": "StepLR",
#     "TYPE": "step",
#     "PARAMS": {
#         "step_size": 10,
#         "gamma": 0.14
#     }
# }

output_dir = "/content/drive/MyDrive/bert4rec/models/rec-transformer-model-10/"

scheduler(data_params=data_params,
          model_params=model_params,
          loggers=loggers,
          optimizer_params=optimizer_params,
          scheduling_parameters=sch_parameters,
          output_dir=output_dir)

# trainer(data_params=data_params,
#         model_params=model_params,
#         loggers=loggers,
#         warmup_steps=True,
#         output_dir=output_dir,
#         modify_last_fc=False,
#         validation=False,
#         optimizer_params=optimizer_params)
