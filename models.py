import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Блок из двух conv 3x3 с BatchNorm.

    residual=True  -> ResNet-блок:  relu(F(x) + shortcut(x))
    residual=False -> plain-блок:   relu(F(x))
    """

    def __init__(self, in_planes, planes, stride=1, residual=True):
        super().__init__()
        self.residual = residual
        self.stride = stride
        # Сколько нулевых каналов нужно дописать к входу, чтобы его можно
        # было сложить с выходом (16 -> 32 или 32 -> 64).
        self.extra_channels = planes - in_planes

        # bias=False: сразу после conv стоит BatchNorm, у которого есть свой
        # сдвиг (beta), так что bias в conv был бы лишним.
        # Первый conv при stride=2 уменьшает картинку вдвое (32->16, 16->8).
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def shortcut(self, x):
        """Приводит вход x к форме выхода блока, не добавляя весов."""
        if self.stride != 1:
            # Уменьшаем размер вдвое, беря каждый второй пиксель.
            x = x[:, :, ::self.stride, ::self.stride]
        if self.extra_channels > 0:
            # Дописываем нулевые каналы в конец: (N, C, H, W) -> (N, C + extra, H, W).
            # Аргументы F.pad идут с последней оси: (W_лево, W_право, H_верх, H_низ, C_начало, C_конец).
            x = F.pad(x, (0, 0, 0, 0, 0, self.extra_channels))
        return x

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            # Главное отличие ResNet: блоку достаточно выучить поправку F(x)
            # к входу, а не всё преобразование целиком.
            out = out + self.shortcut(x)
        # ReLU после сложения, как в оригинальной статье.
        return F.relu(out)


class CifarNet(nn.Module):
    """Сеть глубины 6n+2 для CIFAR-10. residual выбирает ResNet или plain."""

    def __init__(self, depth, residual=True, num_classes=10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "глубина должна быть вида 6n+2: 20, 32, 44, 56, 110..."
        n = (depth - 2) // 6
        self.depth = depth
        self.residual = residual

        # Первый слой: 3 канала RGB -> 16 фильтров, размер 32x32 сохраняется.
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Три стадии. Внутри стадии число фильтров и размер карты постоянны;
        # на входе 2-й и 3-й стадий размер уменьшается вдвое, а фильтров
        # становится вдвое больше (вычислений на слой примерно столько же).
        self.stage1 = self._make_stage(16, 16, n, stride=1)
        self.stage2 = self._make_stage(16, 32, n, stride=2)
        self.stage3 = self._make_stage(32, 64, n, stride=2)

        # Global average pooling: (N, 64, 8, 8) -> (N, 64, 1, 1), затем классификатор.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_planes, planes, n, stride):
        # Только первый блок стадии меняет размер и число каналов,
        # остальные n-1 блоков работают с одинаковой формой.
        blocks = [BasicBlock(in_planes, planes, stride, self.residual)]
        for _ in range(n - 1):
            blocks.append(BasicBlock(planes, planes, 1, self.residual))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        # Инициализация He (Kaiming) для conv, как в статье: сохраняет
        # дисперсию сигнала при прохождении через ReLU-слои.
        # BatchNorm стартует как тождественное преобразование (gamma=1, beta=0).
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # (N, 16, 32, 32)
        x = self.stage1(x)                    # (N, 16, 32, 32)
        x = self.stage2(x)                    # (N, 32, 16, 16)
        x = self.stage3(x)                    # (N, 64, 8, 8)
        x = self.pool(x).flatten(1)           # (N, 64)
        return self.fc(x)                     # (N, 10) -- логиты классов


def resnet(depth):
    """ResNet-20, -32, -44, -56, -110 ..."""
    return CifarNet(depth, residual=True)


def plain(depth):
    """Та же сеть без shortcut-соединений."""
    return CifarNet(depth, residual=False)


if __name__ == "__main__":
    # Быстрая проверка: python models.py
    x = torch.randn(2, 3, 32, 32)
    for depth in (20, 56):
        params = {}
        for name, build in (("plain", plain), ("resnet", resnet)):
            model = build(depth)
            out = model(x)
            # Считаем слои с весами: conv + fc (BatchNorm не считается).
            layers = sum(isinstance(m, (nn.Conv2d, nn.Linear)) for m in model.modules())
            params[name] = sum(p.numel() for p in model.parameters())
            print(f"{name}-{depth}: out {tuple(out.shape)}, weight layers {layers}, "
                  f"params {params[name]:,} ({params[name] / 1e6:.2f}M)")
            assert out.shape == (2, 10)
            assert layers == depth
        # Shortcut без весов -> у plain и ResNet одной глубины параметров поровну.
        # Если не совпадает, значит shortcut добавляет веса (например, conv 1x1).
        assert params["plain"] == params["resnet"], params
    print("OK")
