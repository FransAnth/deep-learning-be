from typing import List, Literal

import torch

from DigitRecognition.ai_resources.ai_classes.mnist_cnn import MnistCnn
from DigitRecognition.ai_resources.ai_classes.neural_network import NeuralNetwork


def load_model(model_type: List[Literal["nn", "cnn"]]):
    if model_type == "nn":
        digit_model = NeuralNetwork()
        digit_model.load_state_dict(
            torch.load(
                f"DigitRecognition/ai_resources/trained_ai_models/nueral_network/neural_network_3HL.pth"
            )
        )

        return digit_model

    elif model_type == "cnn":
        digit_model = MnistCnn()
        digit_model.load_state_dict(
            torch.load(
                f"DigitRecognition/ai_resources/trained_ai_models/mnist_cnn/cnn_for_mnist.pth"
            )
        )

        return digit_model

    else:
        raise (KeyError(f"model_type: {model_type} is invalid."))
