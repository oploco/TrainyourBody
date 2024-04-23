from django.shortcuts import render
from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.shortcuts import get_object_or_404, render
from django.urls import reverse
from django.views import generic

from .mymodel import MyModel

# Create your views here.
def predict_view(request):
    if request.method == 'POST':
        # Obtener los datos de entrada del formulario HTML
        Gender = float(request.POST.get('Gender'))
        Age = float(request.POST['Age'])
        Weight = float(request.POST['Weight'])
        Height = float(request.POST['Height'])
        Duration = float(request.POST['Duration'])
        Heart_Rate = float(request.POST['Heart_Rate'])
        Body_Temp = float(request.POST['Body_Temp'])

        print('ppppp',Age)

        # Crear una instancia de la clase de machine learning
        model = MyModel()

        # Llamar al método predict con los datos de entrada
        result = model.predict(Gender, Age, Weight, Height, Duration, Heart_Rate, Body_Temp)

        # Pasar el resultado a la plantilla HTML
        return render(request, 'index.html', {'result': int(result)})

    return render(request, 'index.html')