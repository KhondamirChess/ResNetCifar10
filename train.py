"""Обучение plain/ResNet на CIFAR-10 по рецепту из статьи (He et al., 2015, 4.2).

Рецепт:
    SGD, momentum 0.9, weight decay 1e-4, батч 128
    lr = 0.1, делим на 10 на 32k и 48k итерациях, стоп на 64k
    (64k итераций при 391 шаге за эпоху -- это ~164 эпохи)

Примеры:
    python train.py --model resnet --depth 20
    python train.py --model plain  --depth 56
    python train.py --model resnet --depth 20 --iters 500   # быстрая проверка

Результаты сохраняются в checkpoints/<model>-<depth>/:
    history.csv -- loss и ошибка по эпохам (для графиков plain vs ResNet)
    last.pt     -- последнее состояние (можно продолжить через --resume)
    best.pt     -- веса с лучшей ошибкой на тесте
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn

from data import get_loaders
from models import plain, resnet


def parse_args():
    p = argparse.ArgumentParser(description="Train plain/ResNet on CIFAR-10")
    p.add_argument("--model", choices=["resnet", "plain"], default="resnet")
    p.add_argument("--depth", type=int, default=20, help="6n+2: 20, 32, 44, 56, 110")
    p.add_argument("--iters", type=int, default=64000, help="всего итераций обучения")
    p.add_argument("--milestones", type=int, nargs=2, default=[32000, 48000],
                   help="итерации, на которых lr делится на 10")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--warmup", type=int, default=0,
                   help="первые N итераций lr=0.01 (в статье так делали для ResNet-110: 400)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints")
    p.add_argument("--resume", action="store_true", help="продолжить с last.pt")
    return p.parse_args()


def lr_at(it, args):
    """Learning rate на итерации it (ступенчатое расписание из статьи)."""
    if it < args.warmup:
        return args.lr / 10
    lr = args.lr
    for m in args.milestones:
        if it >= m:
            lr /= 10
    return lr


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Средний loss и ошибка (%) на всей выборке."""
    model.eval()  # BatchNorm берёт накопленные статистики, а не статистики батча
    total_loss, wrong, n = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        wrong += (logits.argmax(1) != labels).sum().item()
        n += labels.size(0)
    return total_loss / n, 100.0 * wrong / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # cudnn сам подберёт самый быстрый алгоритм свёртки для наших размеров.
    torch.backends.cudnn.benchmark = True

    name = f"{args.model}-{args.depth}"
    run_dir = os.path.join(args.out, name)
    os.makedirs(run_dir, exist_ok=True)

    train_loader, test_loader = get_loaders(args.batch_size, args.workers)
    model = (resnet if args.model == "resnet" else plain)(args.depth).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params:,} параметров, устройство {device}")

    # CrossEntropyLoss = softmax + отрицательный логарифм вероятности верного класса.
    criterion = nn.CrossEntropyLoss()
    # weight decay -- L2-штраф за большие веса, уменьшает переобучение.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.wd)

    it, epoch, best_err = 0, 0, 100.0
    last_path = os.path.join(run_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        it, epoch, best_err = ckpt["iter"], ckpt["epoch"], ckpt["best_err"]
        print(f"продолжаем с итерации {it} (эпоха {epoch})")

    history_path = os.path.join(run_dir, "history.csv")
    fields = ["epoch", "iter", "lr", "train_loss", "train_err", "test_loss", "test_err", "time_s"]
    # При новом запуске начинаем history.csv заново, при --resume дописываем.
    if not (args.resume and os.path.exists(history_path)):
        with open(history_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    # Сколько всего эпох получится (с учётом уже пройденных при --resume).
    total_epochs = epoch + math.ceil((args.iters - it) / len(train_loader))
    while it < args.iters:
        model.train()  # включаем режим обучения (BatchNorm считает статистики по батчу)
        epoch += 1
        start = time.time()
        run_loss, run_wrong, run_n = 0.0, 0, 0

        for images, labels in train_loader:
            if it >= args.iters:
                break
            # lr задаём на каждой итерации: расписание в статье -- по итерациям, а не эпохам.
            lr = lr_at(it, args)
            for group in optimizer.param_groups:
                group["lr"] = lr

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)            # 1. прямой проход: (128, 10)
            loss = criterion(logits, labels)  # 2. насколько ответы неправильные
            optimizer.zero_grad(set_to_none=True)
            loss.backward()                   # 3. градиенты: куда сдвигать каждый вес
            optimizer.step()                  # 4. шаг SGD

            run_loss += loss.item() * labels.size(0)
            run_wrong += (logits.argmax(1) != labels).sum().item()
            run_n += labels.size(0)
            it += 1

        train_loss, train_err = run_loss / run_n, 100.0 * run_wrong / run_n
        test_loss, test_err = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - start

        print(f"эпоха {epoch:3d}/{total_epochs}  iter {it:5d}  lr {lr:.4f}  "
              f"train loss {train_loss:.3f} err {train_err:5.2f}%  "
              f"test loss {test_loss:.3f} err {test_err:5.2f}%  {elapsed:.1f}s")

        with open(history_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, it, lr, f"{train_loss:.4f}", f"{train_err:.2f}",
                                    f"{test_loss:.4f}", f"{test_err:.2f}", f"{elapsed:.1f}"])

        if test_err < best_err:
            best_err = test_err
            torch.save(model.state_dict(), os.path.join(run_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "iter": it, "epoch": epoch, "best_err": best_err, "args": vars(args)},
                   last_path)

    print(f"готово: итоговая ошибка на тесте {test_err:.2f}%, лучшая {best_err:.2f}%")


if __name__ == "__main__":
    main()
