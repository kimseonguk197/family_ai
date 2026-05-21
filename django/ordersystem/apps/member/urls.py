from django.urls import path

from .views import (
    MemberCreateView,
    MemberDeleteView,
    MemberDetailView,
    MemberListView,
    MemberUpdateView,
)

urlpatterns = [
    path("create", MemberCreateView.as_view(), name="member-create"),
    path("detail/<int:pk>", MemberDetailView.as_view(), name="member-detail"),
    path("list", MemberListView.as_view(), name="member-list"),
    path("update/<int:pk>", MemberUpdateView.as_view(), name="member-update"),
    path("delete/<int:pk>", MemberDeleteView.as_view(), name="member-delete"),
]
