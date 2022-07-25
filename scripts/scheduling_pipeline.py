import os
import re
import pandas as pd
from tqdm import trange, tnrange
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from bert4rec_dataset import Bert4RecDataset
from bert4rec_model import RecommendationModel, RecommendationTransformer
from rich.table import Column, Table
from rich import box
from rich.console import Console
from torch import cuda
from train_validate import train_step, validate_step
from sklearn.model_selection import train_test_split
from AttentionTransformer.ScheduledOptimizer import ScheduledOptimizer
from IPython.display import clear_output
from AttentionTransformer.utilities import count_model_parameters
from torch.optim import lr_scheduler as lrs

device = T.device('cuda') if cuda.is_available() else T.device('cpu')


def scheduler(data_params,
              model_params,
              loggers,
              optimizer_params,
              scheduling_parameters,
              output_dir="./models/",
              validation=5):

    console = loggers.get("CONSOLE")

    # tables
    train_logger = loggers.get("TRAIN_LOGGER")
    valid_logger = loggers.get("VALID_LOGGER")

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
    model = RecommendationTransformer(
        vocab_size=model_params.get("VOCAB_SIZE"),
        heads=model_params.get("heads", 4),
        layers=model_params.get("layers", 6),
        emb_dim=model_params.get("emb_dim", 512),
        pad_id=model_params.get("pad_id", 0),
        num_pos=model_params.get("history", 120))

    # model.encoder.sou
    if model_params.get("trained"):
        #   load the already trained model
        console.log("TRAINED MODEL AVAILABLE. LOADING...")
        model.load_state_dict(
            T.load(model_params.get("trained"))["state_dict"])
        console.log("MODEL LOADED")
    console.log(f'MOVING MODEL TO DEVICE: {device}')

    model = model.to(device)

    console.log(
        f"TOTAL NUMBER OF MODEL PARAMETERS: {round(count_model_parameters(model)/1e6, 2)} Million"
    )

    optimizer = getattr(T.optim, optimizer_params.get("OPTIM_NAME"))(
        model.parameters(), **optimizer_params.get("PARAMS"))

    console.log("OPTIMIZER AND MODEL DONE")

    console.log("CONFIGURING DATASET AND DATALOADER")
    console.log("DATA PARAMETERS: ", data_params)
    data = pd.read_csv(data_params.get("path"))
    train_data, valid_data = train_test_split(
        data, test_size=0.25, random_state=model_params.get("SEED"))

    console.log("LEN OF TRAIN DATASET: ", len(train_data))
    console.log("LEN OF VALID DATASET: ", len(valid_data))

    train_dataset = Bert4RecDataset(train_data,
                                    data_params.get("group_by_col"),
                                    data_params.get("data_col"),
                                    data_params.get("train_history", 120),
                                    data_params.get("valid_history", 5),
                                    data_params.get("padding_mode",
                                                    "right"), "train")
    print(f'LEN OF DATASET OBJECT: ', len(train_dataset))
    train_dl = DataLoader(train_dataset,
                          **data_params.get("LOADERS").get("TRAIN"))

    console.save_text(os.path.join(output_dir,
                                   "logs_model_initialization.txt"),
                      clear=True)

    schedule = getattr(lrs, scheduling_parameters.get("SCHEDULER_NAME"))(
        optimizer, **scheduling_parameters.get("PARAMS"))

    losses = []

    for epoch in tnrange(1, model_params.get("EPOCHS") + 1):
        if scheduling_parameters.get("TYPE") == "step":
            schedule.step()
            console.log(f"EPOCH: {epoch} | LR: {schedule.get_lr()[0]}")
        if epoch % 5 == 0:
            clear_output(wait=True)
            console.log(train_logger)
        train_loss, train_acc = train_step(
            model,
            device,
            train_dl,
            optimizer,
            CLIP=model_params.get("CLIP"),
            chunkify=model_params.get("chunkify"))
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
                     optimizer_dict=optimizer.state_dict()),
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
                     optimizer_dict=optimizer.state_dict()),
                os.path.join(output_dir, "model_files",
                             model_params.get("SAVE_STATE_DICT_NAME")))

        losses.append(train_loss)

        console.save_text(os.path.join(output_dir, "logs_training.txt"),
                          clear=True)
        if scheduling_parameters.get("TYPE") == "reduce":
            if scheduling_parameters.get("DECAY_PARAM") == "acc":
                schedule.step(train_acc)
                try:
                    console.log(f"EPOCH: {epoch} | LR: {schedule.get_lr()[0]}")
                except Exception as e:
                    console.log(
                        f"EXCEPTION GETTING LR FROM SCHEDULER: {str(e)}")
                    continue
            else:
                schedule.step(train_loss)

            console.save_text(os.path.join(output_dir, "logs_training.txt"),
                              clear=True)
