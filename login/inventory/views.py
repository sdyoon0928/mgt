from django.shortcuts import render

# Create your views here.

from django.shortcuts import render

def inventory_home(request):
    return render(request, 'inventory/home.html')
