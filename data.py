"""Загрузчики данных CIFAR-10.

Обучение: случайная обрезка 32x32 из картинки, дополненной на 4 пикселя
с каждой стороны, и случайное отражение по горизонтали.
Тест: без аугментаций.
Обе выборки нормализуются средним и std по каналам обучающей выборки CIFAR-10.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Папка, в которой лежит cifar-10-batches-py (скачана вручную).
DATA_ROOT = "data"

# Среднее и стандартное отклонение каждого канала (R, G, B) по 50 000
# обучающим картинкам, в шкале [0, 1]. После нормализации каждый канал
# имеет среднее ~0 и std ~1, это ускоряет и стабилизирует обучение.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms():
    """Возвращает преобразования (train_tf, test_tf)."""
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    train_tf = transforms.Compose([
        # Дополняем картинку нулями до 40x40 и вырезаем случайный кусок 32x32:
        # объект оказывается сдвинут на 0..4 пикселя в любую сторону.
        transforms.RandomCrop(32, padding=4),
        # С вероятностью 0.5 отражаем картинку слева направо.
        transforms.RandomHorizontalFlip(),
        # PIL-картинка (H, W, C) в [0, 255] -> тензор (C, H, W) в [0, 1].
        transforms.ToTensor(),
        normalize,
    ])

    # На тесте аугментаций нет: каждая картинка всегда одна и та же,
    # иначе точность на тесте менялась бы от запуска к запуску.
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, test_tf


def get_loaders(batch_size=128, num_workers=4, root=DATA_ROOT):
    """Создаёт DataLoader'ы для обучения и теста.

    batch_size  -- размер батча (128, как в статье про ResNet).
    num_workers -- число процессов, которые параллельно готовят батчи.
    root        -- папка с датасетом.
    """
    train_tf, test_tf = get_transforms()

    # download=False: датасет уже лежит в data/, ничего не скачиваем.
    # torchvision при загрузке проверяет контрольные суммы файлов.
    train_set = datasets.CIFAR10(root, train=True, download=False, transform=train_tf)
    test_set = datasets.CIFAR10(root, train=False, download=False, transform=test_tf)

    # pin_memory ускоряет копирование батчей на GPU; без GPU он не нужен.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,  # каждую эпоху новый порядок картинок
        num_workers=num_workers,
        pin_memory=pin,
        # Не пересоздавать процессы-загрузчики в начале каждой эпохи.
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,  # порядок на тесте не важен, перемешивать незачем
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


if __name__ == "__main__":
    # Быстрая проверка: python data.py
    train_loader, test_loader = get_loaders()
    images, labels = next(iter(train_loader))
    print("train batch:", tuple(images.shape), "labels:", tuple(labels.shape))
    print("train steps per epoch:", len(train_loader))
    print("test steps:", len(test_loader))

    # Батч: 128 картинок, 3 канала, 32x32 пикселя.
    assert images.shape == (128, 3, 32, 32), images.shape
    # 50 000 / 128 = 390.6 -> 391 шаг (последний батч неполный, 80 картинок).
    assert len(train_loader) == 391, len(train_loader)
    print("OK")
