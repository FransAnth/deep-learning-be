from rest_framework import serializers

from DeepLearning.utils.camel_case_serializer import CamelCaseSerializer

from .models import DigitRecognitionTrainingData


class DigitRecognitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DigitRecognitionTrainingData
        fields = "__all__"
