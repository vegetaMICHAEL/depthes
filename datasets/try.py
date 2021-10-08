import os
import cv2
f = open("3d60_train_copy.txt",'r')
fw = open("3d60_train_try.txt", 'w+')
n=0
for line in f.readlines():
    rgb = line.strip().split(" ")
    #print('/test/depth/data/'+rgb[0])
    
    img = cv2.imread('/test/depth/data/'+rgb[0],cv2.IMREAD_GRAYSCALE)
    if img is None:
        pass
    else:
        fw.write(line)
        print(rgb[0])
        n+=1
    #n+=1
f.close()
print(n)
