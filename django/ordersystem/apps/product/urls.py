from django.urls import path

from .views import (
    ProductCreateView,
    ProductDeleteView,
    ProductDetailView,
    ProductListView,
    ProductUpdateView,
)

urlpatterns = [
    path("create/", ProductCreateView.as_view(), name="product-create"),
    path("detail/<int:pk>/", ProductDetailView.as_view(), name="product-detail"),
    path("detail<int:pk>/", ProductDetailView.as_view(), name="product-detail-compact"),
    path("list/", ProductListView.as_view(), name="product-list"),
    path("update/<int:pk>/", ProductUpdateView.as_view(), name="product-update"),
    path("delete/<int:pk>/", ProductDeleteView.as_view(), name="product-delete"),
]
