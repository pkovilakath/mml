from django.shortcuts import render
from classification.models import Classification
#from classification.serializers import ClassificationSerializer

def index(request):
    tdata=""
    filepath = request.FILES.get('file-upload-train', False)
    if filepath:
        tdata="Got it..."
    classs=Classification.objects.all()
    return render(request, 'index.html',{'tdata':filepath,'classs':classs})