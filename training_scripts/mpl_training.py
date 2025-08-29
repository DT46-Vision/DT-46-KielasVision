# Copyright (c) 2023

import onnx
import os
import torch
import torchvision
from torch import nn


class MLP(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(20 * 28 * 1, 120),  # 输入尺寸 (1,28,20) → 560
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)   # 分类数动态设置
        )

    def forward(self, x):
        return self.layers(x)


def save_model(model, num_classes: int):
    # Save as onnx
    dummy_input = torch.randn(1, 1, 28, 20)  # [batch, channel, H, W]
    torch.onnx.export(model, dummy_input, "mlp.onnx", input_names=["input"], output_names=["output"])

    # Check onnx
    onnx_model = onnx.load("mlp.onnx")
    onnx.checker.check_model(onnx_model)
    print("✅ Exported mlp.onnx, classes =", num_classes)
    print(onnx.helper.printable_graph(onnx_model.graph))


def main():
    # Data transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),         # 转灰度，保证 1 通道
        torchvision.transforms.Resize((28, 20)),    # 固定输入大小 H=28, W=20
        torchvision.transforms.RandomAffine(
            degrees=(-5, 5), translate=(0.08, 0.08), scale=(0.9, 1.1)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomErasing(scale=(0.02, 0.02))
    ])

    # Load dataset
    dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(os.path.dirname(__file__), '..', 'datasets'),
        transform=transform
    )
    print(dataset, "\n")

    # Show label names
    print("classes:\n", dataset.classes, "\n")
    num_classes = len(dataset.classes)

    # Save label names to file for C++ 推理
    with open("labels.txt", "w") as f:
        for name in dataset.classes:
            f.write(name + "\n")
    print("✅ Saved labels.txt")

    # Init model
    model = MLP(num_classes)
    print(model, "\n")

    # Split dataset into train and test (8:2)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)]
    )

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Define data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False)

    # Train and evaluate
    for epoch in range(5):
        model.train()
        for batch, (x, y) in enumerate(train_loader):
            # Forward
            y_pred = model(x)

            # Compute loss
            loss = loss_fn(y_pred, y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            if batch % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch}, Loss: {loss.item():.4f}')

        # Evaluate
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, y in test_loader:
                y_pred = model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            acc = 100 * correct / total
            print(f'Epoch: {epoch}, Accuracy: {acc:.2f}%')

        # Save model on each epoch
        save_model(model, num_classes)
        print("\n")


if __name__ == '__main__':
    main()
