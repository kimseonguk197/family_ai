from rest_framework import generics

from .models import Order, OrderDetail
from .serializers import OrderDetailSerializer, OrderSerializer


class OrderCreateView(generics.CreateAPIView):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer


class OrderDetailView(generics.RetrieveAPIView):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer


class OrderListView(generics.ListAPIView):
    queryset = Order.objects.all().order_by("-id")
    serializer_class = OrderSerializer


class OrderUpdateView(generics.UpdateAPIView):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer


class OrderDeleteView(generics.DestroyAPIView):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer


class OrderDetailCreateView(generics.CreateAPIView):
    queryset = OrderDetail.objects.all()
    serializer_class = OrderDetailSerializer


class OrderDetailDetailView(generics.RetrieveAPIView):
    queryset = OrderDetail.objects.all()
    serializer_class = OrderDetailSerializer


class OrderDetailListView(generics.ListAPIView):
    queryset = OrderDetail.objects.all().order_by("-id")
    serializer_class = OrderDetailSerializer


class OrderDetailUpdateView(generics.UpdateAPIView):
    queryset = OrderDetail.objects.all()
    serializer_class = OrderDetailSerializer


class OrderDetailDeleteView(generics.DestroyAPIView):
    queryset = OrderDetail.objects.all()
    serializer_class = OrderDetailSerializer
