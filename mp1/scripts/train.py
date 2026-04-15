import os

import torch
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import rich
from rich.progress import Progress

from dataset import CaptureDataset
from simple_enet import SimpleENet


# Configurations

##### YOUR CODE STARTS HERE #####

BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
TRAIN_VAL_SPLIT = 0.8
CHECKPOINT_EVERY = 2 #epochs

def loss_fn(y, yp):
    """
    loss function
    
    :param y: torch.Tensor [B, H, W]
    :param yp: torch.Tensor [B, num_classes, H, W]
    """
    loss = F.cross_entropy(yp,y)
    return loss

    
##### YOUR CODE ENDS HERE #####


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = os.path.join("data", "capture")
CHECKPOINT_DIR = os.path.join("data", "checkpoints")
RUNS_DIR = os.path.join("data", "runs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
ds = CaptureDataset(DATASET_PATH)
writer = SummaryWriter(RUNS_DIR)


def visualize(what, writer, images, masks, predictions, step, max_images=4):
    """
    concatenates images for tensorboard logging
    
    :param what: log tag (str)
    :param writer: tensorboard summary writer
    :param images: torch.Tensor [B, 1, H, W]
    :param masks: torch.Tensor [B, 1, H, W]
    :param predictions: torch.Tensor [B, 1, H, W]
    :param step: int
    :param max_images: int
    """
    
    images_rgb = images.repeat(1, 3, 1, 1)
    masks_rgb = masks.repeat(1, 3, 1, 1)
    preds_rgb = predictions.repeat(1, 3, 1, 1)

    data = []
    for i in range(min(max_images, len(images_rgb))):
        data.extend([images_rgb[i], masks_rgb[i], preds_rgb[i]])
    
    grid = torchvision.utils.make_grid(data, nrow=3)
    writer.add_image(f"Comparison/vis-{what}", grid, global_step=step)


def train():
    train_dataset, val_dataset = ds.split(TRAIN_VAL_SPLIT)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SimpleENet().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR)

    step = 0
    
    with Progress() as progress:
        task = progress.add_task("[green]Training ...", total=EPOCHS * len(train_loader))
        for epoch in range(1, EPOCHS + 1):
            model = model.train()

            train_loss = 0
            val_loss = 0

            for (x, y) in train_loader:
                optimizer.zero_grad()
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                yp = model(x)
                loss = loss_fn(y, yp)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                progress.update(task, advance=1)
                step += 1
                writer.add_scalar("Loss/train_batch", loss.item(), step)

            pred = torch.argmax(yp.detach(), dim=1).cpu().unsqueeze(1)
            visualize(
                "train - [image, ground truth, prediction]",
                writer,
                x.detach().cpu(),
                y.detach().cpu().unsqueeze(1),
                pred,
                step
            )

            avg_train_loss = train_loss / len(train_dataset)

            for (x, y) in val_loader:
                with torch.no_grad():
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    yp = model(x)
                    loss = loss_fn(y, yp)
                    val_loss += loss.item()

            pred = torch.argmax(yp.detach(), dim=1).cpu().unsqueeze(1)
            visualize(
                "val - [image, ground truth, prediction]",
                writer,
                x.detach().cpu(),
                y.detach().cpu().unsqueeze(1),
                pred,
                step
            )

            writer.add_scalar("Loss/val", val_loss, step)

            avg_val_loss = val_loss / len(val_dataset)
            
            rich.print(f"Epoch {epoch}: Avg Train Loss: {avg_train_loss:.4f} Avg Val Loss: {avg_val_loss:.4f}")

            if epoch % CHECKPOINT_EVERY == 0:
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"epoch{epoch}.pth"))


if __name__ == '__main__':
    try:
        train()
        rich.print("[green]Training Complete! :o")
    except KeyboardInterrupt:
        rich.print("[red]Training Interrupted! x_X")
    