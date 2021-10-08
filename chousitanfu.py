f = open('./datasets/3d60_val.txt', 'r')
f1 = open('./datasets/sitanfu_val.txt', 'w+')
list = f.readlines()
for l in list:
    dir = l.split('/')
    #print(dir[4])
    if dir[4]=='Stanford2D3D':
        f1.write(l)
        print(l)
