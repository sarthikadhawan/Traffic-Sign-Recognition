import os
from os import path
path1 = r"C:\Users\MyPC\Desktop\IA project Final\\Training"
dirs = os.listdir(path1)
for file in dirs:
    path2 = path1 + "\\" + file
    dirs2 = os.listdir(path2)
    i=1
    for f in dirs2:
        while(os.path.isfile(path2+"\\"+file+"_"+str(i)+".png")):
           print("yes")
           i+=1
        os.rename(path2+"\\"+f,path2+"\\"+file+"_"+str(i)+".png")
        i+=1
