from django.db import models

from apps.member.models import Member
from apps.product.models import Product


class Order(models.Model):
    member = models.ForeignKey(Member, on_delete=models.CASCADE, related_name="orders")
    created_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "order"

    def __str__(self) -> str:
        return f"{self.id} - member:{self.member_id}"


class OrderDetail(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name="details")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="order_details")
    count = models.PositiveIntegerField()

    class Meta:
        db_table = "order_detail"

    def __str__(self) -> str:
        return f"{self.id} - order:{self.order_id} product:{self.product_id}"
