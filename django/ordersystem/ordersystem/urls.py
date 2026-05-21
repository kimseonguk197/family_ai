
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('member/', include('apps.member.urls')),
    path('order/', include('apps.order.urls')),
    path('product/', include('apps.product.urls')),
]
