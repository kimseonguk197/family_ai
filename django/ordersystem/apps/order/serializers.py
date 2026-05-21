from rest_framework import serializers

from .models import Order, OrderDetail


class OrderSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = ["id", "member", "created_time"]
        read_only_fields = ["created_time"]


class OrderDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = OrderDetail
        fields = ["id", "order", "product", "count"]
