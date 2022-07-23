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

import optuna
from optuna.trial import TrialState

device = T.device('cuda') if cuda.is_available() else T.device('cpu')


def tuner(data_params,
          model_params,
          train_params,
          loggers,
          output_dir="./models/"):

    console = loggers.get("CONSOLE")

    train_logger = loggers.get("TRAIN_LOGGER")

    # check if output_dir/model_files available; if not create
    console.log("Verifying if Output Directory for model and logs exists")
    if not os.path.exists(output_dir):
        console.log(f"OUTPUT DIRECTORY DOES NOT EXIST. CREATING...")
        os.mkdir(output_dir)
        console.log(
            f"OUTPUT DIRECTORY FOR MODEL FILES DOES NOT EXIST. CREATING...")
        os.mkdir(os.path.join(output_dir, "model_files"))
        console.log(
            f"OUTPUT DIRECTORY FOR INITIAL MODEL FILES DOES NOT EXIST. CREATING..."
        )
        os.mkdir(os.path.join(output_dir, "model_files_initial"))
    else:
        console.log(f"OUTPUT DIRECTORY EXISTS. CHECKING CHILD DIRECTORY...")
        if not os.path.exists(os.path.join(output_dir, "model_files")):
            console.log(
                f"OUTPUT DIRECTORY FOR MODEL FILES DOES NOT EXIST. CREATING..."
            )
            os.mkdir(os.path.join(output_dir, "model_files"))
            console.log(
                f"OUTPUT DIRECTORY FOR INITIAL MODEL FILES DOES NOT EXIST. CREATING..."
            )
            os.mkdir(os.path.join(output_dir, "model_files_initial"))

    # seed
    console.log(f"SEEDING WITH: {model_params.get('SEED')}")
    T.manual_seed(model_params["SEED"])
    T.cuda.manual_seed(model_params["SEED"])
    T.backends.cudnn.deterministic = True

    console.log("MODEL PARAMS: ", model_params)
    console.log("INITIALIZING MODEL: ", model_params)
    model = RecommendationTransformer(
        vocab_size=model_params.get("VOCAB_SIZE"),
        heads=model_params.get("heads", 4),
        layers=model_params.get("layers", 6),
        emb_dim=model_params.get("emb_dim", 512),
        pad_id=model_params.get("pad_id", 0),
        num_pos=model_params.get("history", 120))

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

    accs = []

    def tune(trial):
        learning_rate = trial.suggest_float("lr",
                                            train_params.get("LR_RANGE")[0],
                                            train_params.get("LR_RANGE")[1],
                                            log=True)
        momentum = trial.suggest_float("momentum",
                                       train_params.get("MOM_RANGE")[0],
                                       train_params.get("MOM_RANGE")[1],
                                       log=True)
        optimizer = T.optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum)
        if len(accs) + 1 % 3 == 0:
            clear_output(wait=True)
            console.log(train_logger)

        train_loss, train_acc = train_step(model, device, train_dl, optimizer,
                                           False, data_params.get("MASK"),
                                           model_params.get("CLIP"),
                                           data_params.get("chunkify"))
        train_logger.add_row(str(len(accs) + 1), str(train_loss),
                             str(train_acc), str(learning_rate), str(momentum))

        console.log(train_logger)

        trial.report(train_acc, len(accs) + 1)

        if len(accs) + 1 > 1 and train_acc > max(accs):
            console.log("SAVING BEST MODEL AT EPOCH -> ", len(accs) + 1)
            console.log("LOSS OF BEST MODEL: ", train_loss)
            console.log("ACCURACY OF BEST MODEL: ", train_acc)
            console.log("LEARNING RATE OF BEST MODEL: ", learning_rate)
            console.log("MOMENTUM OF BEST MODEL: ", momentum)
            T.save(
                model,
                os.path.join(output_dir, "model_files",
                             model_params.get("SAVE_NAME")))
            T.save(
                dict(state_dict=model.state_dict(),
                     epoch=len(accs) + 1,
                     train_acc=train_acc,
                     train_loss=train_loss,
                     optimizer_dict=optimizer._optimizer.state_dict()),
                os.path.join(output_dir, "model_files",
                             model_params.get("SAVE_STATE_DICT_NAME")))

        accs.append(train_acc)
        console.save_text(os.path.join(output_dir, "logs_training.txt"),
                          clear=False)
        # epoch += 1
        return train_acc

    console.log("CREATING STUDY")
    study = optuna.create_study(direction="maximize")
    console.log("CREATED STUDY")
    console.log("STARTED OPTIMIZING")
    study.optimize(tune,
                   n_trials=train_params.get("TRIALS"),
                   timeout=train_params.get("TIMEOUT"))
    console.log("COMPLETED OPTIMIZING")

    pruned_trials = study.get_trials(deepcopy=False,
                                     states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False,
                                       states=[TrialState.COMPLETE])

    console.log("Study statistics: ")
    console.log("  Number of finished trials: ", len(study.trials))
    console.log("  Number of pruned trials: ", len(pruned_trials))
    console.log("  Number of complete trials: ", len(complete_trials))

    console.log("Best trial:")
    trial = study.best_trial

    console.log("  Value: ", trial.value)

    console.log("  Params: ")
    for key, value in trial.params.items():
        console.log("    {}: {}".format(key, value))

    console.save_text(os.path.join(output_dir, "logs_training.txt"),
                      clear=False)
