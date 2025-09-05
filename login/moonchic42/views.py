from django.shortcuts import render
from childabuse.models import ChildObservation
from childabuse.forms import ObservationForm


def home_view(request):
    form = ObservationForm()
    observations = ChildObservation.objects.all().order_by('-observation_date')

    context = {
        'form': form,
        'observation_list': observations,
        'total_kids': observations.count(),
        'danger_kids': observations.filter(is_danger=True).count(),
        'last_report_date': (
            observations.filter(reported=True)
            .order_by('-observation_date')
            .first().observation_date if observations.filter(reported=True).exists() else '없음'
        )
    }

    return render(request, 'main_home.html', context)