import math
import torch
from torch import nn


def training(model, train_dl, device, num_epochs=30):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to(device), data[1].to(device).float()  # Ensure labels are float
            # No need to manually reshape or normalize here if your dataset is correctly prepared
            batch_size = inputs.shape[0]  # Dynamically get the batch size
            inputs = inputs.view(batch_size, 1, 5, 128)
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # For binary classification, apply threshold to output logits
            preds = outputs > 0  # Using 0 as threshold for logits
            #print(preds, labels)
            correct_prediction += (preds.view(-1).long() == labels.long()).sum().item()
            total_prediction += labels.size(0)

        avg_loss = running_loss / len(train_dl)
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        if math.isnan(avg_loss):
            print("Loss is NaN. Stopping training.")
            break

    print('Finished Training')