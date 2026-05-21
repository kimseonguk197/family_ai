from django.urls import path

from .views import (
    OrderCreateView,
    OrderDeleteView,
    OrderDetailCreateView,
    OrderDetailDeleteView,
    OrderDetailDetailView,
    OrderDetailListView,
    OrderDetailUpdateView,
    OrderDetailView,
    OrderListView,
    OrderUpdateView,
)

urlpatterns = [
    path("create/", OrderCreateView.as_view(), name="order-create"),
    path("detail/<int:pk>/", OrderDetailView.as_view(), name="order-detail"),
    path("list/", OrderListView.as_view(), name="order-list"),
    path("update/<int:pk>/", OrderUpdateView.as_view(), name="order-update"),
    path("delete/<int:pk>/", OrderDeleteView.as_view(), name="order-delete"),
    path("detail/create/", OrderDetailCreateView.as_view(), name="order-detail-create"),
    path("detail/detail/<int:pk>/", OrderDetailDetailView.as_view(), name="order-detail-detail"),
    path("detail/list/", OrderDetailListView.as_view(), name="order-detail-list"),
    path("detail/update/<int:pk>/", OrderDetailUpdateView.as_view(), name="order-detail-update"),
    path("detail/delete/<int:pk>/", OrderDetailDeleteView.as_view(), name="order-detail-delete"),
]
