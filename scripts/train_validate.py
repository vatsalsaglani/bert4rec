from tqdm import tqdm, tqdm_notebook
from train_util import *
from torch.autograd import Variable
import torch as T


def train_step(model,
               device,
               loader,
               optimizer,
               scheduled_optim=False,
               MASK=1,
               CLIP=2,
               chunkify=False):
    """Train batch step

    Args:
        model: Model
        device (torch.device): Device to train on 'cuda'/'cpu'
        loader (torch.utils.data.dataloader.DataLoader): DataLoader object
        optimizer: Optimizer
        MASK (int, optional): Mask TOKEN ID. Defaults to 1.

    Returns:
        Tuple(float, float): Epoch/Step Loss, Epoch Accuracy
    """
    model.train()
    total_loss = 0
    total_counts = 0
    train_accs = []
    train_bs = []
    for _, batch in enumerate(tqdm_notebook(loader, desc="Train Loader"),
                              4 if chunkify else 1):

        source = Variable(batch["source"].to(device))
        source_mask = Variable(batch["source_mask"].to(device))
        target = Variable(batch["target"].to(device))
        target_mask = Variable(batch["target_mask"].to(device))
        train_bs += [source.size(0)]
        mask = target_mask == MASK

        optimizer.zero_grad()

        output = model(source, source_mask)

        loss = calculate_loss(output, target, mask)

        total_counts += 1
        total_loss += loss.item()

        loss.backward()
        if CLIP:
            T.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        if scheduled_optim:
            optimizer.step_and_update_lr()
        else:
            optimizer.step()

        mean = calculate_accuracy(output, target, mask)
        train_accs += [mean.item()]

    epoch_acc = calculate_combined_mean(train_bs, train_accs)

    return total_loss / total_counts, epoch_acc.item()


def validate_step(model, loader, device, MASK=1):
    """Validation Step

    Args:
        model: Model
        loader (torch.utils.data.dataloader.DataLoader): Dataloader
        device (torch.device): Device to validate on
        MASK (int, optional): Mask TOKEN ID. Defaults to 1.

    Returns:
        Tuple(float, float): Epoch Validation Loss, Epoch Validation Accuracy
    """
    model.eval()
    total_loss = 0
    total_counts = 0
    valid_accs = []
    valid_bs = []

    for _, batch in enumerate(tqdm_notebook(loader, desc="Valid Loader")):

        source = Variable(batch["source"].to(device))
        source_mask = Variable(batch["source_mask"].to(device))
        target = Variable(batch["target"].to(device))
        target_mask = Variable(batch["target_mask"].to(device))
        mask = target_mask == MASK

        output = model(source, source_mask)

        loss = calculate_loss(output, target, mask)
        total_counts += 1
        total_loss += loss.item()

        mean = calculate_accuracy(output, target, mask)
        valid_accs += [mean.item()]
        valid_bs += [source.size(0)]

    epoch_acc = calculate_combined_mean(valid_bs, valid_accs)

    return total_loss / total_counts, epoch_acc.item()
