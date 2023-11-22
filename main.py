from tkinter import filedialog
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import  *
from PIL import Image
from tkinter import *
from tkinter import ttk
import tkinter
import psycopg2
from datetime import date
import time
import torch
import cvzone
from matplotlib import pyplot as plt
from tensorflow import keras
from imgaug import augmenters as iaa 
import imgaug as ia
import imageio as io
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
import glob
#model=YOLO('yolov8n.pt')
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8n.yaml').load('yolov8n.pt') 
my_date=[]
my_time=[]
analysis=[]
#result = model.train(data='coco128.yaml', epochs=3, imgsz=640) 
#result=model.train(data='mnist160', epochs=100, imgsz=64)
curr_time = time.strftime("%H:%M:%S", time.localtime())
today = date.today()
connection = psycopg2.connect(database="yolo8",
                        host="localhost",
                        user="postgres",
                        password="1234",
                        port="5432")
cursor = connection.cursor()
#currenttime = datetime.datetime.now()
#datetimetk=Tk()
#wt=700
#ht=150
#swt=datetimetk.winfo_screenwidth()
#sht=datetimetk.winfo_screenheight()
#xt=int((swt-wt)/2)
#yt=int((sht-ht)/2)
#datetimetk.geometry(f'{wt}x{ht}+{xt}+{yt}') 
#root.overrideredirect(True)
#atetimetk.config(background = 'grey')
#datetimetk.title('Select Date')
#datetimetk.destroy
root = Tk()
w=700
h=250
sw=root.winfo_screenwidth()
sh=root.winfo_screenheight()
x=int((sw-w)/2)
y=int((sh-h)/2)
root.geometry(f'{w}x{h}+{x}+{y}')
#root.overrideredirect(True)
root.config(background = 'grey')
root.title('YOLO8 MODEL')
frm = ttk.Frame(root,padding=10,relief='raised')
frm.grid()
def analyze_data(start_date,end_date):
    mysql="SELECT * FROM report WHERE date >= "+"\'" +start_date +"\'"+" AND date <= "+"\'" +end_date+"\'" 
   
    cursor.execute(mysql)
    c=[1,2,3,4,5,6,7,8,9,10]
    d=[1,2,3,4,5,6,7,8,9,10]
   
    mysql_fetch=cursor.fetchall()
    for x in range(0,len(mysql_fetch)):
       my_date.append(mysql_fetch[x][0])
       
       my_time.append(mysql_fetch[x][1])
       #c.append(d+int(X))
       analysis.append(mysql_fetch[x][2])
    
    plt.plot(my_date,analysis,marker='*')
    plt.xlabel("Date & Time")
    plt.ylabel("Number of poeple")
    plt.show()
#img = io.imread("i3.jpg")
#ia.imshow(image=img)
def  augimg():
    images = []
    images_path = glob.glob("C:\\Users\\megop\\OneDrive\\Desktop\\poeple detection\\Counting-people-in-a-marathon-using-YOLOv8-main\\my_preview\\i4.jpg")
    for img_path in images_path:
        img = cv2.imread(img_path)
        images.append(img)
    seq = iaa.Sequential([
    		iaa.Crop(px=(0, 16)),
    		iaa.Fliplr(0.5),
    		iaa.GaussianBlur(sigma=(0, 3.0)),
            
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
               rotate=(-30, 30),
               scale=(0.5, 1.5)),
    # 3. Multiply
            iaa.Multiply((0.8, 1.2)),
    # 4. Linearcontrast
            iaa.LinearContrast((0.6, 1.4)),
    # Perform methods below only sometimes
            iaa.Sometimes(0.5,
        # 5. GaussianBlur
            iaa.GaussianBlur((0.0, 3.0))
        )

            
            ])
    while True:
            augmented_images =seq(images=images)
            for my_img in augmented_images:
                cv2.imshow("Image", my_img)
                #mon_img=cv2.imread(my_img)
                mon_img = np.random.randint(255, size=(300, 600, 3))
                cv2.waitKey(0)
                
                cv2.imwrite("C:\\Users\\megop\\OneDrive\\Desktop\\poeple detection\\Counting-people-in-a-marathon-using-YOLOv8-main\\my_preview\\my-image"+str(x)+".jpg",my_img)

    #for batch_idx in range(1000):
    		#images = load_batch(batch_idx)
    		#images_aug = seq(images=images)

def augementImages():
     datagen = ImageDataGenerator(rotation_range=30,        
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
 fill_mode='nearest')
     img = load_img('i3.jpg')
     x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
     x = x.reshape((1,) + x.shape) 
     i = 0
     for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely

def browseFiles():
        
        
        # Change label contents
        global filename
         
        filename = filedialog.askopenfilename(initialdir = "/",
                                            title = "Select a File",
                                            filetypes = (("Video files",
                                                            "*.mp4 .wmv .flv .mp3  .vob .ogv .ogg .drc .webm  .mng .avi .mov .qt .yuv .rm .rmvb .viv .asf .amv .svi .3gp .3g2 .mxf .roq .nsv .f4v .f4p .f4a .f4b*"),
                                                            ("Image files",
                                                            "*.jpg .png .jpeg .tiff .gif .gifv .psd*"),
                                                        ))
        label_file_explorer.configure(text=filename)
label_file_explorer = Label(root,
                            text = "Enter the image or the vedio you want to test",
                            width = 100, height = 4,
                            fg = "blue",background='grey')

      
button_explore = ttk.Button(root,
                        text = "Browse Files",
                        command =browseFiles)
label_file_explorer.grid(column = 4, row = 0)
  
button_explore.grid(column = 4, row = 1)
#ttk.Label(frm, text="",background='grey').grid(column=0, row=0)
def test_model():
    result = model.train(data='coco128.yaml', epochs=5, imgsz=640) 
def printpres(): 
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        print(metrics.top1)   # top1 accuracy
        print(metrics.top5) 
def calc_precision():
            prescion=Tk()
            my_wt=700
            my_ht=150
            sc_wt=prescion.winfo_screenwidth()
            sc_ht=prescion.winfo_screenheight()
            my_xt=int((sc_wt-my_wt)/2)
            my_yt=int((sc_ht-my_ht)/2)
            value_inside = tkinter.StringVar(prescion)
            prescion.geometry('%dx%d+%d+%d'%(my_wt,my_ht,my_xt,my_yt))
            prescion.config(background='grey')
            prescion.title('Calculate Precision')
            value_inside.set("Select an Option")
            start_date = Label(prescion,
                        text = "TP").place(x = 110,
                                                y = 30) 
            my_list=ttk.Entry(prescion,width=30)
            my_list.place(x=110,y=60)
            
            end_date = Label(prescion,
                        text = "FP").place(x = 340,
                                                y = 30) 
            my_list2=ttk.Entry(prescion,width=30)
            my_list2.place(x=340,y=60)
            my_btn=ttk.Button(prescion,text='Calculate',command=printpres).place(x=280,y=100) 

def window():
    c=[1,2,3,4,5,6,7,8,9,10]
    datetimetk=Tk()
    wt=700
    ht=150
    swt=datetimetk.winfo_screenwidth()
    sht=datetimetk.winfo_screenheight()
    xt=int((swt-wt)/2)
    yt=int((sht-ht)/2)
    datetimetk.geometry(f'{wt}x{ht}+{xt}+{yt}')
    datetimetk.title('Select Date Range')
    datetimetk.config(background = 'grey')
    my_frame=ttk.Frame(datetimetk,padding=10,relief='raised')
    value_inside = tkinter.StringVar(datetimetk)
  
# Set the default value of the variable
    value_inside.set("Select an Option")
    start_date = Label(datetimetk,
                  text = "Start Date").place(x = 110,
                                           y = 30) 
    my_list=ttk.Entry(datetimetk,width=30)
    my_list.place(x=110,y=60)
    
    end_date = Label(datetimetk,
                  text = "End Date").place(x = 340,
                                           y = 30) 
    my_list2=ttk.Entry(datetimetk,width=30)
    my_list2.place(x=340,y=60)
    def analytic():
        analyze_data(my_list.get(),my_list2.get())
        
    my_btn=ttk.Button(datetimetk,text='Analyze Data',command=analytic).place(x=280,y=100)
  
    my_frame.grid()
ttk.Button(root, text="Choose File", command=browseFiles).grid(column=4, row=1)
ttk.Button(root, text="OK", command=root.destroy).grid(column=4, row=2)
ttk.Button(root, text="Analyze Data", command=window).grid(column=4, row=3)
ttk.Button(root, text="Calculate Presicion", command=calc_precision).grid(column=4, row=4)
ttk.Button(root, text="Train Model", command=test_model).grid(column=4, row=5)
ttk.Button(root, text="Augument Image", command=augimg).grid(column=4, row=6)
root.mainloop()
cy1=383
rect_list = []
offset=4
poeple_count=[]
counter=[]

    #root.destroy()


        

    



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
#file_path=input('Enter your file path: ')        
#myfile=open(file_path)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
#print(browseFiles)
if filename.endswith('.mp4') or filename.endswith('.wmv') or filename.endswith('.flv') or filename.endswith('.mp3') or filename.endswith('.vob') or filename.endswith('.ogv') or filename.endswith('.ogg') or filename.endswith('.drc') or filename.endswith('.webm') or filename.endswith('.mng') or  filename.endswith('.mov') or  filename.endswith('.avi') or  filename.endswith('.qt') or  filename.endswith('.yuv') or  filename.endswith('.rm') or  filename.endswith('.rmvb') or  filename.endswith('.viv') or  filename.endswith('.asf') or  filename.endswith('.amv') or  filename.endswith('.svi') or  filename.endswith('.3gp') or  filename.endswith('.3g2') or  filename.endswith('.mxf') or  filename.endswith('.roq') or  filename.endswith('.nsv') or  filename.endswith('.f4v') or  filename.endswith('.f4p') or  filename.endswith('.f4a') or  filename.endswith('.f4b'):
    cap=cv2.VideoCapture(filename)
elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.tiff') or filename.endswith('.gif') or filename.endswith('.gifv') or filename.endswith('.psd'):
    cap=cv2.imread(filename)

my_file = open("C:\\Users\\megop\OneDrive\\Desktop\\poeple detection\\Counting-people-in-a-marathon-using-YOLOv8-main\\coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()
area1=[(0,0),(0,490),(1174,569),(1174,-300)]

#area2=[(279,392),(250,397),(423,477),(454,469)]
c = set() 
while True: 
       
    if filename.endswith('.mp4') or filename.endswith('.wmv') or filename.endswith('.flv') or filename.endswith('.mp3') or filename.endswith('.vob') or filename.endswith('.ogv') or filename.endswith('.ogg') or filename.endswith('.drc') or filename.endswith('.webm') or filename.endswith('.mng') or  filename.endswith('.mov') or  filename.endswith('.avi') or  filename.endswith('.qt') or  filename.endswith('.yuv') or  filename.endswith('.rm') or  filename.endswith('.rmvb') or  filename.endswith('.viv') or  filename.endswith('.asf') or  filename.endswith('.amv') or  filename.endswith('.svi') or  filename.endswith('.3gp') or  filename.endswith('.3g2') or  filename.endswith('.mxf') or  filename.endswith('.roq') or  filename.endswith('.nsv') or  filename.endswith('.f4v') or  filename.endswith('.f4p') or  filename.endswith('.f4a') or  filename.endswith('.f4b'):
        ret,frame = cap.read()
    elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg') or filename.endswith('.tiff') or filename.endswith('.gif') or filename.endswith('.gifv') or filename.endswith('.psd'):
        ret=cap
        frame=cap
    if filename.endswith('.mp4') or filename.endswith('.wmv') or filename.endswith('.flv') or filename.endswith('.mp3') or filename.endswith('.vob') or filename.endswith('.ogv') or filename.endswith('.ogg') or filename.endswith('.drc') or filename.endswith('.webm') or filename.endswith('.mng') or  filename.endswith('.mov') or  filename.endswith('.avi') or  filename.endswith('.qt') or  filename.endswith('.yuv') or  filename.endswith('.rm') or  filename.endswith('.rmvb') or  filename.endswith('.viv') or  filename.endswith('.asf') or  filename.endswith('.amv') or  filename.endswith('.svi') or  filename.endswith('.3gp') or  filename.endswith('.3g2') or  filename.endswith('.mxf') or  filename.endswith('.roq') or  filename.endswith('.nsv') or  filename.endswith('.f4v') or  filename.endswith('.f4p') or  filename.endswith('.f4a') or  filename.endswith('.f4b'):
        if not ret:
            break
    else:
        pass
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,500))
    #cv2.polylines(frame,[np.array(area,np.int32)],True,(0,255,0),2)
    
  

   
   

   
    results=model.predict(frame)
    #print(results)
   
    a=results[0].boxes.data
#    print(a)
    px=pd.DataFrame(a).astype("float")
    #print(model)
    list=[] 
   
        
   # for index,row in px.iterrows():
#        print(row)
    points = [] 
   
    for index,row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        n=class_list[int(row[5])]
        #b=class_list[int(row[3])]
        #print(b)
        #print(c.count('person'))
        if 'person' in n:
             points.append([x1,y1,x2,y2]) 
            #x=str(a[0][7])
             cv2.rectangle(frame,(x1,y1),(x2,y2),(139,0,0),2)
             cv2.putText(frame,str(n),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
             canvas = np.zeros(frame.shape, np.uint8)
             img2gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

             ret,thresh = cv2.threshold(img2gray,128,255,cv2.THRESH_BINARY_INV)
             im2=cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
             contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

             print(len(contours))

             for cont in contours:
                cv2.drawContours(canvas, cont, -1, (0, 255, 0), 3)

            
            #sql='INSERT INTO report (date,time,analysis) VALUES (%s,%s,%s)'
            #val=(today,curr_time,x)
            #cursor.execute(sql,val)
           # number_of_objects_in_image=c.count('person')
            #print ("The number of objects in this image: ", str(number_of_objects_in_image))
             connection.commit()
             print()

            #print(cursor.rowcount, "record inserted.")
        boxes_id = tracker.update(points)
        
        for box_id in boxes_id:
                x3,y3,x4,y4,id=box_id
                results=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
              #  cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),1)
               # cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,0),2)
                if results>=0:
                    c.add((x4,y4))
                    cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                    cv2.circle(frame,(x4,y4),5,(0,255,0),-1)
                    cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

        
                
         
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    #cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
   


    print(len(rect_list))
    u=(len(c))
    print(c)
    cvzone.putTextRect(frame, f'people count:-{u}', (50,60),2,2)
    #cv2.imshow("RGB", frame)
   
 
          
    sql='INSERT INTO report (date,time,analysis) VALUES (%s,%s,%s)'
    val=(today,curr_time,u)
    cursor.execute(sql,val)
    
    connection.commit()
    print(cursor.rowcount, "record inserted.")
    #cv2.putText(frame,'number of people is ='+str(boxes_id),(50,65),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
    cv2.imshow("My video ", frame)
    
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

