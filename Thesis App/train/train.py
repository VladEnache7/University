import torch
from torch import nn
from tqdm import tqdm
from metrics import ObjectDetectionAccuracy


def train_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_function: ObjectDetectionAccuracy, device: torch.device) -> tuple[float, float]:
    model.train()
    train_loss, train_acc = 0, 0
    accuracy_function.reset()
    for batch, (X, y) in enumerate(dataloader):
        y_pred = model(X)
        loss_dict = loss_fn(y_pred, y)
        weight_dict = loss_fn.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k]
                     for k in loss_dict.keys() if k in weight_dict)

        train_loss += losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        accuracy_function.update(y_pred, y)
    train_loss /= len(dataloader)
    train_acc = accuracy_function.compute()
    accuracy_function.reset()
    return train_loss, train_acc


def test_epoch(model: nn.Module, dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, accuracy_function: ObjectDetectionAccuracy, device: torch.device) -> tuple[float, float]:
    model.eval()
    test_loss, test_acc = 0, 0
    accuracy_function.reset()
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss_dict = loss_fn(y_pred, y)
            weight_dict = loss_fn.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k]
                         for k in loss_dict.keys() if k in weight_dict)

            test_loss += losses.item()
            accuracy_function.update(y_pred, y)
        test_loss /= len(dataloader)

        test_acc = accuracy_function.compute()
        accuracy_function.reset()
    return test_loss, test_acc


def train(model: nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, accuracy_function: ObjectDetectionAccuracy, device: torch.device, epochs: int) -> dict[str, list[float]]:
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_epoch(model=model, dataloader=train_dataloader, loss_fn=loss_fn,
                                            optimizer=optimizer, accuracy_function=accuracy_function, device=device)
        test_loss, test_acc = test_epoch(model=model, dataloader=test_dataloader,
                                         loss_fn=loss_fn, accuracy_function=accuracy_function, device=device)
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_loss)
        results["train_acc"].append(train_acc)
        results["test_acc"].append(test_acc)
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

    return results
