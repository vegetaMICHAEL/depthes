import os
f = open("3d60_train.txt",'r')
fw = open("3d60_train_g.txt", 'w+')
n=0
for line in f.readlines():
    rgb = line.strip().split(" ")
    #print('/test/depth/data/'+rgb[0])
    if os.path.exists('/test/depth/data/'+rgb[0]):
        fw.write(line)
        n+=1
    else:
       pass
       #print(line)
    #n+=1
f.close()
print(n)
