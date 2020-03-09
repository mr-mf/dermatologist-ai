from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
import torch


class Loader:

    __use_cuda = torch.cuda.is_available()

    async def transform_input(self, image):
        """
        Takes in the image and applies transformations to it:
        -- re sizes the image to 224 x 224
        -- normalises mean and std of all RGB layers (standardized for resnet)
        -- converts to PyTorch tensor

        :param image: FileField which contains an image
        :return: pytorch tensor representation of the image
        """
        img = Image.open(BytesIO(image))
        transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        image_tensor = transformation(img)

        if self.__use_cuda:
            image_tensor.cuda()

        # un-squeeze the first dimension so we can have the batch information
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    async def transform_output(self, tensor_array):
        """
        Detaches from GPU and converts to a numpy array

        :param tensor_array:
        :return:
        """
        if self.__use_cuda:
            tensor_array.cpu()
        # convert to numpy
        numpy_array = tensor_array.detach().numpy()
        # remove the first dimension and get the first element out
        melanoma_probs = numpy_array.squeeze(0)[0]
        return melanoma_probs

