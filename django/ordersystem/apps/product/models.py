from django.db import models


class Product(models.Model):
    name = models.CharField(max_length=150)
    price = models.DecimalField(max_digits=10, decimal_places=2)
    stock_quantity = models.PositiveIntegerField(default=0)

    class Meta:
        db_table = "product"

    def __str__(self) -> str:
        return f"{self.id} - {self.name}"
