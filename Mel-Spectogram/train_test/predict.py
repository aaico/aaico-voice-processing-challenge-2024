import torch

def predict(model, test, device):
    model.eval()
    with torch.no_grad():
        for data in test:
            if torch.all(data == 0.):
                prediction = 1
            else:
                # Get the input features and target labels, and put them on the GPU
                inputs = data.to(device)
                
                batch_size = inputs.shape[0]  # Dynamically get the batch size
                inputs = inputs.view(batch_size, 1, 5, 128)
                # Normalize the inputs
                inputs_m, inputs_s = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_m) / inputs_s

                # Get predictions
                outputs = model(inputs)

                probs = torch.sigmoid(outputs)  # Convert logits to probabilities
                prediction = int(probs.round().item())  # Round probabilities to get binary predictions
        #print('pred', prediction)

    return prediction
