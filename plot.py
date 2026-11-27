"""Графики ошибки по итерациям, как рис. 6 в статье He et al., 2015.

Слева plain-20 и plain-56, справа ResNet-20 и ResNet-56.
Пунктир -- ошибка на обучающих картинках, сплошная линия -- на тестовых.

Запуск:
    python plot.py                               # берёт checkpoints_server/, сохраняет fig6.png
    python plot.py --dir checkpoints --out my.png
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")  # рисуем сразу в файл, окно на экране не нужно
import matplotlib.pyplot as plt

DEPTHS = (20, 56)
COLORS = {20: "#1f77b4", 56: "#d62728"}  # синий -- 20 слоёв, красный -- 56


def read_history(path):
    """Читает history.csv и возвращает столбцы: итерации, ошибка train, ошибка test."""
    iters, train_err, test_err = [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            iters.append(int(row["iter"]))
            train_err.append(float(row["train_err"]))
            test_err.append(float(row["test_err"]))
    return iters, train_err, test_err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="checkpoints_server", help="папка с <model>-<depth>/history.csv")
    p.add_argument("--out", default="fig6.png")
    args = p.parse_args()

    # Два графика рядом с общей осью Y, чтобы plain и ResNet было легко сравнить.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, model, title in zip(axes, ("plain", "resnet"), ("Plain", "ResNet")):
        for depth in DEPTHS:
            path = os.path.join(args.dir, f"{model}-{depth}", "history.csv")
            if not os.path.exists(path):
                print(f"нет файла {path}, пропускаю")
                continue
            iters, train_err, test_err = read_history(path)
            # Ось X в десятках тысяч итераций, как в статье (1e4).
            x = [i / 1e4 for i in iters]
            color = COLORS[depth]
            ax.plot(x, train_err, "--", color=color, linewidth=1, alpha=0.8)
            ax.plot(x, test_err, "-", color=color, linewidth=2,
                    label=f"{title}-{depth} (тест {test_err[-1]:.2f}%)")
            print(f"{model}-{depth}: ошибка в конце -- train {train_err[-1]:.2f}%, test {test_err[-1]:.2f}%")

        # Моменты, когда lr делится на 10 (32k и 48k итераций).
        for m in (3.2, 4.8):
            ax.axvline(m, color="gray", linewidth=0.6, linestyle=":")

        ax.set_title(title)
        ax.set_xlabel("итерации (×10⁴)")
        ax.set_xlim(0, 6.4)
        ax.set_ylim(0, 20)  # как в статье: от 0 до 20%
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    axes[0].set_ylabel("ошибка (%)")
    fig.suptitle("CIFAR-10: пунктир -- обучение, сплошная -- тест")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"график сохранён в {args.out}")


if __name__ == "__main__":
    main()
