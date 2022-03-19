from tqdm import tqdm, tqdm_notebook
from train_util import *
from torch.autograd import Variable


def train_step(model, device, loader, optimizer, MASK=1):
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
    for _, batch in enumerate(tqdm_notebook(loader, desc="Train Loader")):

        source = Variable(batch["source"].to(device))
        target = Variable(batch["target"].to(device))
        train_bs += [source.size(0)]
        mask = source == MASK

        optimizer.zero_grad()

        output = model(source)

        loss = calculate_loss(output, target, mask)

        total_counts += 1
        total_loss += loss.item()

        loss.backward()
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
        target = Variable(batch["target"].to(device))
        mask = source == MASK

        output = model(source)

        loss = calculate_loss(output, target, mask)
        total_counts += 1
        total_loss += loss.item()

        mean = calculate_accuracy(output, target, mask)
        valid_accs += [mean.item()]
        valid_bs += [source.size(0)]

    epoch_acc = calculate_combined_mean(valid_bs, valid_accs)

    return total_loss / total_counts, epoch_acc.item()
