import torch
import os
from torchvision import models
import torch.nn as nn
import torch.nn.functional as f
from src.main.loader import Loader

model_path = '../model/'
model = os.path.join(model_path, 'resnet50.pt')


class Predictor:
    __model = None
    __use_cuda = torch.cuda.is_available()
    __loader = Loader()

    @classmethod
    async def __get_model(cls):
        """
        Load the model file from the model directory
        """
        if cls.__model is None:
            # load in the untrained resent architecture
            cls.__model = models.resnet50()
            # add the final layer to the model
            cls.__model.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 3)
            )
            # load in the model state dictionary
            cls.__model.load_state_dict(torch.load(model))
            # put the model in the evaluation state
            cls.__model.eval()

            if cls.__use_cuda:
                cls.__model.cuda()

    @classmethod
    async def predict(cls, input):
        """
        Make a prediction with the resnet 50
        """
        await cls.__get_model()
        input_torch = await cls.__loader.transform_input(input)
        output_torch = cls.__model(input_torch)
        # convert scores to probabilities
        output_torch_probs = f.softmax(output_torch, dim=1)
        # convert to numpy
        output_np = await cls.__loader.transform_output(output_torch_probs)
        return output_np
