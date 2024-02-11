from model import model
from train_test.preprocessing import get_train_test
from train_test.training import training
from train_test.inference import inference

import torch

# Create the model and put it on the GPU if available
audioModel = model.AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
audioModel = audioModel.to(device)

# Check that it is on Cuda
next(audioModel.parameters()).device

train_dl, val_dl = get_train_test()
training(audioModel, train_dl, device)

# uncomment to save the model
# torch.save(audioModel.state_dict(), 'model/model.pkl')
# changing model mode to eval
audioModel.eval()

# running validation set
inference(audioModel, val_dl, device)