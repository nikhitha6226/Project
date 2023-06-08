from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class drink_driving_detection(models.Model):

    idnumber= models.CharField(max_length=300)
    City_Location= models.CharField(max_length=30000)
    day= models.CharField(max_length=30000)
    Sex= models.CharField(max_length=30000)
    Age= models.CharField(max_length=30000)
    Time= models.CharField(max_length=30000)
    Day_of_week= models.CharField(max_length=30000)
    Educational_level= models.CharField(max_length=30000)
    Vehicle_driver_relation= models.CharField(max_length=30000)
    Driving_experience= models.CharField(max_length=30000)
    Type_of_vehicle= models.CharField(max_length=30000)
    Owner_of_vehicle= models.CharField(max_length=30000)
    Ser_year_of_veh= models.CharField(max_length=30000)
    Lanes_or_Medians= models.CharField(max_length=30000)
    Road_allignment= models.CharField(max_length=30000)
    Road_surface_type= models.CharField(max_length=30000)
    Vehicle_movement= models.CharField(max_length=30000)
    Prediction= models.CharField(max_length=30000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



