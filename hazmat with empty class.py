import collections

import torch, torchvision
from torchvision.models import resnet50, ResNet50_Weights

BATCH_SIZE = 32

# (mac only)
torch.set_default_device(torch.device("mps"))

# Set the random seed
torch.manual_seed(37)

weights = ResNet50_Weights.DEFAULT
# This is the actual model
res = resnet50(weights=weights)

def get_dataloaders(
        *, batch_size: int, train_proportion: float
) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """load the data, and prepare it to be fed into the model."""
    data_path = "Hazmat_Individual"
    # This may be overly complicated, but would allow you to put together
    #   more than one transform to be applied to the dataset.
    transform = torchvision.transforms.Compose([
        weights.transforms(), # Preprocessing function
    ])
    full_dataset = torchvision.datasets.ImageFolder(
        data_path, transform=transform
    )
    # Apple Silicon
    generator = torch.Generator(device="mps").manual_seed(37)
    # No Apple Silicon
    # generator = torch.Generator().manual_seed(37)
    # Nvidia GPU
    # generator = torch.Generator(device="cuda").manual_seed(37)
    train_set, valid_set = torch.utils.data.random_split(
        full_dataset,
        [train_proportion, 1-train_proportion],
        generator = generator,
    )
    train = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    validation = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)
    return train, validation

train, validation = get_dataloaders(batch_size=BATCH_SIZE, train_proportion=0.7)
print(f"Train Batches: {len(train)}")
print(f"Validation Batches: {len(validation)}")

# print(res.children)
res.fc = torch.nn.Identity()

# Don't train any of the ResNet weights
for param in res.parameters():
    param.requires_grad=False

model = torch.nn.Sequential(collections.OrderedDict([
        ('resnet', res),
        ('final', torch.nn.Linear(in_features=2048, out_features=15)),
        ('softmax', torch.nn.Softmax(dim=1)),
]))
# print(model)

lr = 0.0001
print(f"Initial learning rate: {lr:.4f}")

# choose optimizer
optimizer = torch.optim.Adam(model.final.parameters(), lr=lr)
# Set the model to training mode
model.train()
# Choose a loss function
loss_fn = torch.nn.CrossEntropyLoss()

# This sets the number of epochs
for i in range(150):
    batch_losses = []
    batch_accuracies = []
    print(f"=== Epoch {i+1} ===")
    for (image_batch, label_batch) in train:
        preds = model(image_batch)
        loss = loss_fn(preds, label_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_losses.append(float(loss))
        accuracy = int(sum(preds.argmax(1) == label_batch))/len(label_batch)
        batch_accuracies.append(accuracy)
        cur_loss = sum(batch_losses)/len(batch_losses)
        cur_accuracy = sum(batch_accuracies)/len(batch_accuracies)
        print("Train:", end="\t\t")
        print(f"Batch: {len(batch_losses)}", end="\t")
        print(f"Loss: {cur_loss:.4f}", end="\t")
        print(f"Accuracy: {cur_accuracy:.4f}", end="\r")
    # Finished training for this epoch.  Time to validate.
    print()
    batch_losses = []
    batch_accuracies = []
    for (image_batch, label_batch) in validation:
        with torch.no_grad():
            preds = model(image_batch)
            loss = loss_fn(preds, label_batch)
            batch_losses.append(float(loss))
            accuracy = int(sum(preds.argmax(1) == label_batch))/len(label_batch)
            batch_accuracies.append(accuracy)
            cur_loss = sum(batch_losses)/len(batch_losses)
            cur_accuracy = sum(batch_accuracies)/len(batch_accuracies)
            print("Validation:", end="\t")
            print(f"Batch: {len(batch_losses)}", end="\t")
            print(f"Loss: {cur_loss:.4f}", end="\t")
            print(f"Accuracy: {cur_accuracy:.4f}", end="\r")
    print()

print("Training complete.")

# save the  completed model
torch.save(model.state_dict(), "hazmat_weights_individual.pth")
print("Model saved as 'hazmat_weights_individual.pth'.")