import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time

model=YOLO("yolov8s.pt")

def RGB(event,x,y,flags,param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsRGB=[x,y]
        print(colorsRGB)


cv2.namedWindow('RED')
cv2.setMouseCallback('RED', RGB)

cap=cv2.VideoCapture('highway.mp4')

with open ("classeslist.txt",'r') as my_file:
    list_class=my_file.read().split('\n')

tracker= Tracker()
cy1=200
cy2= 350
offset =10
cars_down={}
counter=[]
cars_up= {}
counter1 =[]

fourcc=cv2.VideoWriter_fourcc (*'XVID')
out= cv2.VideoWriter('output.avi',fourcc,20.0,(1020,637))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 637))

    # Performing prediction using the YOLO model
    results = model.predict(frame)
    a =results[0].boxes.data
    px= pd.DataFrame(a).astype("float")
    bbox_list =[]

    for _, row in px.iterrows():
        x1 = int(row[0])
        y1 =int(row[1])
        x2= int(row[2])
        y2= int(row[3])
        d= int(row[5])
        c =list_class[d]
        if 'car' in c:
            w =x2 - x1
            h= y2 - y1
            bbox_list.append([x1, y1, w, h])

    bbox_id =tracker.update(bbox_list)

    for bbox in bbox_id:
        x1, y1, w, h, id = bbox
        x2=x1 + w
        y2=y1 + h
        cx =(x1 + x2) // 2
        cy =(y1 + y2) // 2
        cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 0, 255), 2)

        
        if cy1 - offset<cy < cy1 +offset:
            if id not in cars_down:
                cars_down[id]=time.time()
            cv2.circle(frame, (cx, cy),4,(0, 255, 0), -1)
            cv2.putText(frame, f'ID: {id}',(x1, y1 - 10),cv2.FONT_HERSHEY_COMPLEX, 0.6,(255, 255, 255), 1)
        
        if id in cars_down and cy2 - offset<cy<cy2+offset:
            elapsed_time=time.time() - cars_down[id]
            if id not in counter:
                counter.append(id)
                distance=20  # meters
                speed_ms=distance/elapsed_time
                speed_kh=speed_ms*3.6
                cv2.putText(frame, f'{int(speed_kh)} Km/h', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        
        if cy2 - offset<cy < cy2+offset:
            if id not in cars_up:
                cars_up[id]=time.time()
            cv2.circle(frame,(cx, cy), 4,(0, 0, 255), -1)
            cv2.putText(frame, f'ID: {id}',(x1,y1- 10),cv2.FONT_HERSHEY_COMPLEX, 0.6,(255, 255, 255), 1)
        
        if id in cars_up and cy1 - offset < cy < cy1 + offset:
            elapsed1_time=time.time()-cars_up[id]
            if id not in counter1:
                counter1.append(id)
                distance1 = 20  # meters
                speed_ms1 =distance1 / elapsed1_time
                speed_kh1= speed_ms1 * 3.6
                cv2.putText(frame,f'{int(speed_kh1)} Km/h',(x2, y2),cv2.FONT_HERSHEY_COMPLEX, 0.8,(0, 255, 255),2)

    
    cv2.line (frame,(100, cy1),(frame.shape[1]-100,cy1), (255,255, 255),2)
    cv2.putText(frame,'L1',(5,cy1- 5),cv2.FONT_HERSHEY_COMPLEX,0.8,(0, 255, 255), 2)
    cv2.line(frame,(50, cy2),(frame.shape[1]-50, cy2), (255, 255, 255), 2)
    cv2.putText(frame, 'L2',(5,cy2 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,255, 255), 2)

    
    d =len(counter)
    u= len(counter1)
    cv2.putText(frame, f'Going Down:{d}',(60,90),cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,255, 255),2)
    cv2.putText(frame,f'Going Up: {u}', (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255,255), 2)

    
    out.write(frame)

    cv2.imshow("RGB",frame)
    if cv2.waitKey(1) &0xFF== 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


