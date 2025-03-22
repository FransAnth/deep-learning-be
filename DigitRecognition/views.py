from django.shortcuts import render
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DigitRecognitionTrainingData
from .serializers import DigitRecognitionSerializer

# Create your views here.


class DigitRecogApiView(APIView):
    def get(self, request):
        digit_recog_query_set = DigitRecognitionTrainingData.objects.all()
        serializer = DigitRecognitionSerializer(digit_recog_query_set, many=True)

        return Response(serializer.data)

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


class DigitDataset(APIView):
    def get(self, request):

        dataset_info = []
        # Getting count of dataset for each Digit
        for i in range(10):
            count = DigitRecognitionTrainingData.objects.filter(label=i).count()
            dataset_info.append({"label": i, "count": count})

        return Response(dataset_info)
