from rest_framework import serializers
from classification.models import Classification
class ClassificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Classification
        #fields = ('id', 'name', 'email', 'message')
        fields = '__all__'