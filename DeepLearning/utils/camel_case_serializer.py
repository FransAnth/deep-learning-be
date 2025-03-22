import inflection
from rest_framework import serializers


class CamelCaseSerializer(serializers.ModelSerializer):
    """
    Converts request data keys from camelCase to
    snake_case and response keys from snake_case to
    camelCase
    """

    def to_internal_value(self, data):
        # Convert request keys from camelCase to snake_case
        snake_case_data = {
            inflection.underscore(key): value for key, value in data.items()
        }

        return super().to_internal_value(snake_case_data)

    def to_representation(self, instance):
        # Convert response data keys from snake_case to camelCase
        camel_case_data = {
            inflection.camelize(key, False): value
            for key, value in super().to_representation(instance).items()
        }

        return camel_case_data
