import torch

def inference (model, val_dl, device):
  correct_prediction = 0
  total_prediction = 0

  # Disable gradient updates
  with torch.no_grad():
    for data in val_dl:
      # Get the input features and target labels, and put them on the GPU
      inputs, labels = data[0].to(device), data[1].to(device)

      labels = labels.long()
      batch_size = inputs.shape[0]  # Dynamically get the batch size
      inputs = inputs.view(batch_size, 1, 5, 128)
      # Normalize the inputs
      inputs_m, inputs_s = inputs.mean(), inputs.std()
      inputs = (inputs - inputs_m) / inputs_s

      # Get predictions
      outputs = model(inputs)

      probs = torch.sigmoid(outputs)  # Convert logits to probabilities
      prediction = probs.round()  # Round probabilities to get binary predictions
      #print(prediction)
      # Count of predictions that matched the target label
      correct_prediction += (prediction.squeeze().long() == labels).sum().item()
      total_prediction += labels.size(0)
    
  acc = correct_prediction/total_prediction
  print(f'Accuracy: {acc:.2f}, Total items: {total_prediction}')