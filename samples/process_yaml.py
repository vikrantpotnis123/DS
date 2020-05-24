#!/usr/bin/python3.6
import yaml

with open(r'fruits.yaml') as file:
    fruits_list = yaml.load(file, Loader=yaml.FullLoader)
    print(fruits_list)
     
