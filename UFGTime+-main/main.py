import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from argument import PROJECT_ROOT, get_args
from src.data_provider import data_provider
from src.model import UFGTimePlus
from src.utils import cheb_approx, evaluate, get_frame, set_seed


def align_target(forecast, target, long_pred):
    target = target.permute(0, 2, 1).contiguous()
    if long_pred:
        return forecast[:, :, -1], target[:, :, -1]
    return forecast, target


@torch.no_grad()
def validate(model, loader, device, long_pred):
    model.eval()
    total_loss, total_elements = 0.0, 0
    for x, y in loader:
        forecast, y = align_target(model(x.float().to(device)), y.float().to(device), long_pred)
        total_loss += nn.functional.mse_loss(forecast, y).item() * y.numel()
        total_elements += y.numel()
    return total_loss / total_elements


@torch.no_grad()
def test(model, loader, device, long_pred):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        forecast, y = align_target(model(x.float().to(device)), y.float().to(device), long_pred)
        preds.append(forecast.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds).astype(np.float64, copy=False)
    trues = np.concatenate(trues).astype(np.float64, copy=False)
    mape, mae, rmse = evaluate(trues, preds)
    return dict(mse=float(np.mean((preds - trues) ** 2)), mae=float(mae),
                rmse=float(rmse), mape=float(mape))


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training configs: {args}")
    train_set, train_loader = data_provider(args, "train")
    _, val_loader = data_provider(args, "val")
    _, test_loader = data_provider(args, "test")
    if not 1 <= args.k < train_set.num_ts:
        raise ValueError(f"k must be between 1 and {train_set.num_ts - 1} external neighbors")
    approx = np.array([cheb_approx(frame, args.cheb_order) for frame in get_frame("Haar")])
    model = UFGTimePlus(seq_length=args.seq_len, signal_length=args.signal_len,
                        pred_length=args.pred_len, hidden_size=args.hidden_size,
                        embed_size=args.embed_size, num_ts=train_set.num_ts,
                        device=device, approx=approx, s=args.s, lev=1, num_topk=args.k,
                        knn_dist="cosine", exclude_self=True,
                        window_centering=(args.window_centering == "mean")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.decay_rate,
                                 betas=(args.beta1, args.beta2), eps=1e-8)
    scheduler = None
    if args.scheduler_patience is not None:
        scheduler = ReduceLROnPlateau(optimizer, "min", patience=args.scheduler_patience)
    loss_fn = nn.MSELoss()
    best_val, selected_epoch, metrics = float("inf"), 0, None
    for epoch in range(1, args.train_epochs + 1):
        start = time.time()
        model.train()
        total_loss, total_elements = 0.0, 0
        for x, y in train_loader:
            x, y = x.float().to(device), y.float().to(device)
            optimizer.zero_grad()
            forecast, y = align_target(model(x), y, args.long_pred)
            loss = loss_fn(forecast, y)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
            total_elements += y.numel()
        val_loss = validate(model, val_loader, device, args.long_pred)
        if not np.isfinite(val_loss):
            raise RuntimeError(f"Non-finite validation loss at epoch {epoch}")
        if scheduler is not None:
            scheduler.step(val_loss)
        print(f"| Epoch {epoch:3d} | time {time.time() - start:.2f}s "
              f"| train {total_loss / total_elements:.9f} | val {val_loss:.9f} |")
        if args.checkpoint_selection == "best_val" and val_loss <= best_val:
            best_val, selected_epoch = val_loss, epoch
            metrics = test(model, test_loader, device, args.long_pred)
    if args.checkpoint_selection == "last":
        selected_epoch = args.train_epochs
        metrics = test(model, test_loader, device, args.long_pred)
    if metrics is None:
        raise RuntimeError("Training completed without an evaluation")
    line = (f"dataset={args.run_name} seed={args.seed} selection={args.checkpoint_selection} "
            f"epoch={selected_epoch} MSE={metrics['mse']:.9f} MAE={metrics['mae']:.9f} "
            f"RMSE={metrics['rmse']:.9f} MAPE={metrics['mape']:.9%}")
    print(line)
    result_dir = PROJECT_ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    with (result_dir / f"{args.run_name}.txt").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


if __name__ == "__main__":
    main()
