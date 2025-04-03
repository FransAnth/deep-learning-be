import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from DigitRecognition.utils import load_model

from .models import DigitRecognitionTrainingData
from .serializers import DigitRecognitionSerializer

# initialize the model
model_type = "cnn"
digit_model = load_model(model_type)

print(f"Model loaded for {model_type}")
digit_model.eval()


class DigitRecogApiView(APIView):

    def post(self, request):
        digit_pixels = request.data.get("pixels")

        # convert string of pixels to tensors
        pixels = [float(x) for x in digit_pixels.split(",")]
        pixel_tensor = torch.tensor(pixels, dtype=torch.float)

        if model_type == "nn":
            input_tensor = pixel_tensor.view(-1, 28 * 28)
        elif model_type == "cnn":
            input_tensor = pixel_tensor.view(1, 28, 28)

        output = digit_model(input_tensor)
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
