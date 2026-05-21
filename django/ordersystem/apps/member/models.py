from django.db import models


class Member(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255)

    class Meta:
        db_table = "member"

    def __str__(self) -> str:
        return f"{self.id} - {self.name}"
