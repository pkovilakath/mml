from django.shortcuts import render
from datetime import datetime
import os


def home(request):
    if request.method == "POST" and request.FILES['file']:
        handle_uploaded_file(request.FILES['file'], str(request.FILES['file']))
    return render(request, "app/home.html",
                  {
                      'number': 5,
                      'mainTitle': 'ML - Classification ',
                      'year': datetime.now().year,
                  }
                  )


def handle_uploaded_file(file, filename):
    if not os.path.exists('upload/'):
        os.mkdir('upload/')

    with open('upload/' + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)