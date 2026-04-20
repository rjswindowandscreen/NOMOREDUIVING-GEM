import torch
import numpy as np
import cv2
from simple_enet import SimpleENet


##### YOUR CODE STARTS HERE #####
# DO NOT CHANGE ANY FUNCTION HEADERS

# load your best model
def load_model() -> SimpleENet:
    path_to_your_model = "data/checkpoints/epoch50.pth"
    model = SimpleENet()
    model.load_state_dict(torch.load(path_to_your_model, weights_only=True))
    return model

def inference(model: SimpleENet, image: np.ndarray, device: str) -> np.ndarray:
    """
    The main image processing pipeline for your model
    
    :param model: pytorch model
    :type model: SimpleENet
    :param image: a BGR image taken from the GEM's camera
    :type image: np.ndarray
    :param device: the device on which the model should run on ("cpu" or "cuda")
    :type device: str
    :return: binary lane-segmented image
    :rtype: ndarray
    """
    pred = None
    og_image = image
    image = cv2.resize(image, (640, 384)) #resize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   #grayscale
    image_3D = np.expand_dims(image, axis=0)   # add channel dimension
    image_4D = np.expand_dims(image_3D, axis=0)   # add batch dimension    
    tensor = torch.from_numpy(image_4D).float()/ 255.0
    tensor = tensor.to(device)
    
    j=0
    model.eval()
    with torch.no_grad():
        pred = model(tensor)
        prediction_output = torch.argmax(pred, dim=1)
        prediction_output = prediction_output.squeeze(0)
        mask = prediction_output
        pred = mask.cpu().numpy().astype(np.uint8)


    
    
    masked_pred = cv2.resize(pred, (og_image.shape[1], og_image.shape[0]))

    return masked_pred


##### YOUR CODE ENDS HERE #####