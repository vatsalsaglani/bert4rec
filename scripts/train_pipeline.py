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

device = T.device('cuda') if cuda.is_available() else T.device('cpu')


def trainer(data_params,
            model_params,
            loggers,
            optimizer_params=None,
            warmup_steps=False,
            output_dir="./models/",
            modify_last_fc=False,
            validation=5):

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

    if modify_last_fc:

        new_word_embedding = nn.Embedding(model_params.get("NEW_VOCAB_SIZE"),
                                          model_params.get("emb_dim"), 0)
        new_word_embedding.weight.requires_grad = False
        console.log(
            f"REQUIRES GRAD for `NEW WORD EMBEDDING` set to {new_word_embedding.weight.requires_grad}"
        )

        new_word_embedding.weight[:model.encoder.word_embedding.weight.size(
            0)] = model.encoder.word_embedding.weight.clone().detach()

        model.encoder.word_embedding = new_word_embedding
        # model.encoder.word_embedding.weight.retain_grad()
        console.log(
            f"WORD EMBEDDING MODIFIED TO `{model.encoder.word_embedding}`")

        # console.log(
        #     f"WORD EMBEDDING REQUIRES GRAD `{model.encoder.word_embedding.weight.requires_grad}`"
        # )
        model.encoder.word_embedding.weight.requires_grad = True
        new_lin_layer = nn.Linear(model_params.get("emb_dim"),
                                  model_params.get("NEW_VOCAB_SIZE"))
        new_lin_layer.weight.requires_grad = False
        new_lin_layer.weight[:model.lin_op.weight.
                             size(0)] = model.lin_op.weight.clone().detach()
        model.lin_op = new_lin_layer
        # model.lin_op.weight.retain_grad()
        model.lin_op.weight.requires_grad = True
        console.log("MODEL LIN OP: ", model.lin_op.out_features)

    model = model.to(device)

    console.log(
        f"TOTAL NUMBER OF MODEL PARAMETERS: {round(count_model_parameters(model)/1e6, 2)} Million"
    )

    # if warmup_steps:
    #     optimizer = ScheduledOptimizer(
    #         T.optim.SGD(params=model.parameters(),
    #                     lr=model_params.get("LEARNING_RATE"),
    #                     momentum=0.8,
    #                     nesterov=True), 1e-6, model_params.get("emb_dim"))
    # else:
    optim_name = optimizer_params.get("OPTIM_NAME")
    if optim_name == "SGD":
        optimizer = T.optim.SGD(params=model.parameters(),
                                **optimizer_params.get("PARAMS"))
    elif optim_name == "ADAM":
        optimizer = T.optim.Adam(params=model.parameters(),
                                 **optimizer_params.get("PARAMS"))

    else:
        optimizer = T.optim.SGD(params=model.parameters(),
                                lr=model_params.get("LEARNING_RATE"),
                                momentum=0.8,
                                nesterov=True)

    if warmup_steps:
        optimizer = ScheduledOptimizer(optimizer, 1e-6,
                                       model_params.get("emb_dim"))

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
    train_dl = DataLoader(train_dataset,
                          **data_params.get("LOADERS").get("TRAIN"))
    # if validation:
    #     valid_dataset = Bert4RecDataset(
    #         valid_data, data_params.get("group_by_col"),
    #         data_params.get("data_col"), data_params.get("train_history", 120),
    #         data_params.get("valid_history", 5),
    #         data_params.get("padding_mode", "right"), "valid")
    console.save_text(os.path.join(output_dir,
                                   "logs_model_initialization.txt"),
                      clear=True)
    # if full_train:
    #     train_dl = DataLoader(train_dataset + valid_dataset,
    #                           **data_params.get("LOADERS").get("TRAIN"))
    # else:
    # train_dl = DataLoader(train_dataset,
    #                       **data_params.get("LOADERS").get("TRAIN"))

    # if validation:
    #     valid_dl = DataLoader(valid_dataset,
    #                           **data_params.get("LOADERS").get("VALID"))

    # if full_train:
    #     train_dl += valid_dl

    losses = []
    for epoch in tnrange(1, model_params.get("EPOCHS") + 1):
        # train_dataset = Bert4RecDataset(
        #     train_data, data_params.get("group_by_col"),
        #     data_params.get("data_col"), data_params.get("train_history", 120),
        #     data_params.get("valid_history", 5),
        #     data_params.get("padding_mode", "right"), "train")
        # train_dl = DataLoader(train_dataset,
        #                       **data_params.get("LOADERS").get("TRAIN"))
        if epoch % 3 == 0:
            clear_output(wait=True)
        train_loss, train_acc = train_step(model, device, train_dl,
                                           optimizer, warmup_steps,
                                           data_params.get("MASK"),
                                           model_params.get("CLIP"),
                                           data_params.get("chunkify"))
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

        if validation and epoch > 1 and epoch % validation == 0:
            valid_dataset = Bert4RecDataset(
                valid_data, data_params.get("group_by_col"),
                data_params.get("data_col"),
                data_params.get("train_history", 120),
                data_params.get("valid_history", 5),
                data_params.get("padding_mode", "right"), "valid")
            valid_dl = DataLoader(valid_dataset,
                                  **data_params.get("LOADERS").get("VALID"))
            valid_loss, valid_acc = validate_step(model, valid_dl, device,
                                                  data_params.get("MASK"))

            valid_logger.add_row(str(epoch), str(valid_loss), str(valid_acc))
            console.log(valid_logger)

            del valid_dataset, valid_dl

            console.log("VALIDATION DONE AT EPOCH ", epoch)

            console.save_text(os.path.join(output_dir, "logs_training.txt"),
                              clear=True)
        console.save_text(os.path.join(output_dir, "logs_training.txt"),
                          clear=True)
