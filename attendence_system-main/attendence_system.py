import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import Image
from IPython.display import HTML
import sys
import math
import glob
import os
import urllib
import csv
from datetime import datetime
import face_recognition
import argparse
import imutils
import dlib

video=cv2.VideoCapture(0)

person1=face_recognition.load_image_file('person1.jpeg')
person1_encoding=face_recognition.face_encodings(person1)[0]

person2=face_recognition.load_image_file('person2.jpeg')
person2_encoding=face_recognition.face_encodings(person2)[0]

person3=face_recognition.load_image_file('person3.jpeg')
person3_encoding=face_recognition.face_encodings(person3)[0]

person4=face_recognition.load_image_file('person4.jpeg')
person4_encoding=face_recognition.face_encodings(person4)[0]

known_face_encoding=[
    person1_encoding,
    person2_encoding,
    person3_encoding,
    person4_encoding
    ]
known_faces_names=[
    'person1',
    'person2',
    'person3',
    'person4'
    ]
students=known_faces_names.copy()
face_locations=[]
face_encodings=[]
face_names=[]
s=True
now=datetime.now()
current_date=now.strftime('%Y-%m-%d')

f=open('current_date.csv','r+',newline='')
lnwriter=csv.writer(f)
while True:
    _,frame=video.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=small_frame[:,:,::-1]
    if s:
        face_locations=face_recognition.face_locations(rgb_small_frame)
        face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names=[]
        for face_encodings in face_encodings:
            matches=face_recognition.compare_faces(known_face_encoding,face_encodings)
            name=''
            face_distance=face_recognition.face_distance(known_face_encoding,face_encodings)
            best_match_index=np.argmin(face_distance)
            if matches[best_match_index]:
                name=known_faces_names[best_match_index]
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time=now.strftime('%Y-%m-%d')
                    lnwriter.writerow([name,current_time])
    cv2.imshow('attendence system',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
video.release()
cv2.destroyAllWindows()
f.close()
