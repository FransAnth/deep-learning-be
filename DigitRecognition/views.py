import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from DigitRecognition.ai_resources.nn_classes.neural_network import NeuralNetwork

from .models import DigitRecognitionTrainingData
from .serializers import DigitRecognitionSerializer

# initialize the model
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10  # Digits 0-9
hidden_layer_count = 3

digit_model = NeuralNetwork(input_size, hidden_size, output_size, hidden_layer_count)
digit_model.load_state_dict(
    torch.load(
        f"DigitRecognition/ai_resources/trained_ai_models/neural_network_{hidden_layer_count}HL.pth"
    )
)
print("model_loaded")
digit_model.eval()


class DigitRecogApiView(APIView):

    def post(self, request):
        digit_pixels = request.data.get("pixels")

        # convert string of pixels to tensors
        pixels = [float(x) for x in digit_pixels.split(",")]
        pixel_tensor = torch.tensor(pixels, dtype=torch.float)

        # plt.gray()
        # plt.imshow(pixel_tensor.view(28, 28), interpolation="nearest")
        # plt.axis("off")  # Remove axes for a clean image
        # plt.savefig("digit.png", dpi=300, bbox_inches="tight")

        output = digit_model(pixel_tensor.view(-1, 28 * 28))
        confidence, prediction = torch.max(
            output, 1
        )  # Get the highest probability class

        print("Confidense", confidence)
        return Response(prediction[0].item())


class DigitDataset(APIView):
    def get(self, request):

        dataset_info = []
        # Getting count of dataset for each Digit
        for i in range(10):
            count = DigitRecognitionTrainingData.objects.filter(label=i).count()
            dataset_info.append({"label": i, "count": count})

        return Response(dataset_info)

    def post(self, request):
        """Expected Json:
        data = [
            {
                "label": "the digit label",
                "features": "comma-separated features"
            },
            ....
        ]
        """

        if not isinstance(request.data, list):  # Ensure it's a list of objects
            return Response(
                {"error": "Expected a list of objects"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        serializer = DigitRecognitionSerializer(data=request.data, many=True)
        if serializer.is_valid():
            objects = [
                DigitRecognitionTrainingData(**data)
                for data in serializer.validated_data
            ]  # Create instances
            DigitRecognitionTrainingData.objects.bulk_create(objects)

            return Response(serializer.data, status=status.HTTP_200_OK)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
