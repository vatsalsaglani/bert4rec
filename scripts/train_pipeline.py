import os
import re
import pandas as pd
from tqdm import trange, tnrange
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert4rec_dataset import Bert4RecDataset
from bert4rec_model import RecommendationModel
from rich.table import Column, Table
from rich import box
from rich.console import Console
from torch import cuda
from train_validate import train_step, validate_step
from sklearn.model_selection import train_test_split
from AttentionTransformer.ScheduledOptimizer import ScheduledOptimizer

device = T.device('cuda') if cuda.is_available() else T.device('cpu')


def trainer(data_params,
            model_params,
            loggers,
            warmup_steps=False,
            output_dir="./models/"):

    # console instance

    console = loggers.get("CONSOLE")

    # tables
    train_logger = loggers.get("TRAIN_LOGGER")
    valid_logger = loggers.get("VALID_LOGGER")

    # check if output_dir/model_files available; if not create
    if not os.path.exists(output_dir):
        console.log(f"OUTPUT DIRECTORY DOES NOT EXIST. CREATING...")
        os.mkdir(output_dir)
        os.mkdir(os.path.join(output_dir, "model_files"))
        os.mkdir(os.path.join(output_dir, "model_files_initial"))
    else:
        console.log(f"OUTPUT DIRECTORY EXISTS. CHECKING CHILD DIRECTORY...")
        if not os.path.exists(os.path.join(output_dir, "model_files")):
            os.mkdir(os.path.join(output_dir, "model_files"))
            os.mkdir(os.path.join(output_dir, "model_files_initial"))

    # seed
    console.log("SEED WITH: ", model_params.get("SEED"))
    T.manual_seed(model_params["SEED"])
    T.cuda.manual_seed(model_params["SEED"])
    T.backends.cudnn.deterministic = True

    # intialize model
    console.log("MODEL PARAMS: ", model_params)
    console.log("INITIALIZING MODEL: ", model_params)
    model = RecommendationModel(vocab_size=model_params.get("VOCAB_SIZE"),
                                heads=model_params.get("heads", 4),
                                layers=model_params.get("layers", 6),
                                emb_dim=model_params.get("emb_dim", 512),
                                pad_id=model_params.get("pad_id", 0),
                                num_pos=120)
    if model_params.get("trained"):
        #   load the already trained model
        console.log("TRAINED MODEL AVAILABLE. LOADING...")
        model.load_state_dict(
            T.load(model_params.get("trained"))["state_dict"])
        console.log("MODEL LOADED")
    console.log(f'MOVING MODEL TO DEVICE: {device}')
    model = model.to(device)

    if warmup_steps:
        optimizer = ScheduledOptimizer(
            T.optim.Adam(params=model.parameters(),
                         lr=model_params.get("LEARNING_RATE"),
                         betas=(0.9, 0.98),
                         eps=1e-9), 1e-6, model_params.get("emb_dim"))
    else:
        optimizer = T.optim.Adam(params=model.parameters(),
                                 lr=model_params.get("LEARNING_RATE"))

    console.log("OPTIMIZER AND MODEL DONE")

    console.log("CONFIGURING DATASET AND DATALOADER")
    console.log("DATA PARAMETERS: ", data_params)
    data = pd.read_csv(data_params.get("path"))
    train_data, valid_data = train_test_split(
        data, test_size=0.25, random_state=model_params.get("SEED"))

    train_dataset = Bert4RecDataset(train_data,
                                    data_params.get("group_by_col"),
                                    data_params.get("data_col"),
                                    data_params.get("train_history", 120),
                                    data_params.get("valid_history", 5),
                                    data_params.get("padding_mode",
                                                    "right"), "train")
    valid_dataset = Bert4RecDataset(valid_data,
                                    data_params.get("group_by_col"),
                                    data_params.get("data_col"),
                                    data_params.get("train_history", 120),
                                    data_params.get("valid_history", 5),
                                    data_params.get("padding_mode",
                                                    "right"), "valid")

    train_dl = DataLoader(train_dataset,
                          **data_params.get("LOADERS").get("TRAIN"))
    valid_dl = DataLoader(valid_dataset,
                          **data_params.get("LOADERS").get("VALID"))

    losses = []
    for epoch in tnrange(1, model_params.get("EPOCHS") + 1):

        train_loss, train_acc = train_step(model, device, train_dl, optimizer,
                                           data_params.get("MASK"))
        train_logger.add_row(str(epoch), str(train_loss), str(train_acc))

        console.log(train_logger)

        if epoch == 1:
            console.log(f"Saving Initial Model")
            T.save(
                model,
                os.path.join(output_dir, "model_files_initial",
                             model_params.get("SAVE_NAME")))
            T.save(
                dict(state_dict=model.state_dict(),
                     epoch=epoch,
                     train_loss=train_loss,
                     train_acc=train_acc,
                     optimizer_dict=optimizer._optimizer.state_dict()
                     if warmup_steps else optimizer.state_dict()),
                os.path.join(output_dir, "model_files_initial",
                             model_params.get("SAVE_STATE_DICT_NAME")))

        if epoch > 1 and min(losses) > train_loss:
            console.log("SAVING BEST MODEL AT EPOCH -> ", epoch)
            console.log("LOSS OF BEST MODEL: ", train_loss)
            console.log("ACCURACY OF BEST MODEL: ", train_acc)
            T.save(
                model,
                os.path.join(output_dir, "model_files",
                             model_params.get("SAVE_NAME")))
            T.save(
                dict(state_dict=model.state_dict(),
                     epoch=epoch,
                     train_acc=train_acc,
                     train_loss=train_loss,
                     optimizer_dict=optimizer._optimizer.state_dict()
                     if warmup_steps else optimizer.state_dict()),
                os.path.join(output_dir, "model_files",
                             model_params.get("SAVE_STATE_DICT_NAME")))

        losses.append(train_loss)

        valid_loss, valid_acc = validate_step(model, valid_dl, device,
                                              data_params.get("MASK"))

        valid_logger.add_row(str(epoch), str(valid_loss), str(valid_acc))
        console.log(valid_logger)

        console.save_text(os.path.join(output_dir, "logs.txt"))