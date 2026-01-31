import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet_classifier import build_resnet18
from utils.dataset import get_train_dataloader, get_val_dataloader


def train_model(num_epochs=10,
                batch_size=32,
                learning_rate=1e-4,
                save_path='resnet_face_classifier.pth',
                pretrained=False,
                freeze_backbone=False,
                num_workers=0):
    """
    Train the ResNet18 classifier.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = build_resnet18(num_classes=2, pretrained=pretrained)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    train_loader = get_train_dataloader(batch_size=batch_size, num_workers=num_workers)
    val_loader = get_val_dataloader(batch_size=batch_size, num_workers=num_workers)

    best_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"Model saved with validation accuracy: {val_accuracy:.2f}%")

    print("Training completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet18 classifier")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="resnet_face_classifier.pth")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                save_path=args.save_path,
                pretrained=args.pretrained,
                freeze_backbone=args.freeze_backbone,
                num_workers=args.num_workers)
