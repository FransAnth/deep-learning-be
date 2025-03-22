from django.db import models
from django.utils import timezone


# Create your models here.
class DigitRecognitionTrainingData(models.Model):
    id = models.AutoField(primary_key=True)
    label = models.CharField(max_length=1, default=None)
    features = models.TextField(default=None)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        db_table = "digit_recognition_training_data"
