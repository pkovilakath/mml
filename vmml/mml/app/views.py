from django.shortcuts import render
from datetime import datetime


def home(request):

    return render(request, "app/home.html",
                  {
                      'number': 5,
                      'mainTitle': 'ML - Classification ',
                      'year': datetime.now().year,
                   }
                  )
