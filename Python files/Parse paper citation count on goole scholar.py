# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 14:03:11 2016

@author: anam
"""

import scholarly

import csv
from random import randint
from time import sleep
#import requests


f = open("paper_title1.csv", 'r')

    

try:
    reader = csv.reader(f)
    for row in reader:
        #print (str(row[1]))
        #s=requests.Session()
        #s.cookies.clear()
        sleep(randint(1,2))
        a = next(scholarly.search_pubs_query((row[1])),None)
        #print(a)
        try:
            print(a.citedby)
            dr = [row[0],a.citedby]
        except :
            dr = [row[0],0]
            print(0)
        with open(r'computer_cite.csv', 'a') as g:
            writer = csv.writer(g)
            writer.writerow(dr)
finally:
    f.close()
    #g.close()
