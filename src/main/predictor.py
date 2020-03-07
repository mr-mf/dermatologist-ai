import torch

model_path = '../model/'


class Predictor:
    model = None
    use_cuda = torch.cuda.is_available()

    @classmethod
    async def get_model(cls):
        """
        Load the model file from the model directory
        """
        if cls.model is None:
            cls.model.load_state_dict(torch.load('resnet50.pt'))

            if cls.use_cuda:
                cls.model.cuda()

        return cls.model

    @classmethod
    async def predict(cls, input):
        """
        Make a prediction with the resnet 50
        """
        model = cls.get_model()
        return model(input)

