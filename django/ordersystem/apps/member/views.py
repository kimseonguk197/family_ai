from rest_framework import generics

from .models import Member
from .serializers import MemberSerializer


class MemberCreateView(generics.CreateAPIView):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer


class MemberDetailView(generics.RetrieveAPIView):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer


class MemberListView(generics.ListAPIView):
    queryset = Member.objects.all().order_by("-id")
    serializer_class = MemberSerializer


class MemberUpdateView(generics.UpdateAPIView):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer


class MemberDeleteView(generics.DestroyAPIView):
    queryset = Member.objects.all()
    serializer_class = MemberSerializer
