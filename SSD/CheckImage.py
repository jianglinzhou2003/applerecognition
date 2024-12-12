from scipy.misc import imread
import os


root = "E:/FInal_Data"
files = os.listdir(root)
count = 0
for file in files:
    if file.endswith("txt"):
        continue
    count += 1
    print(count)
    filename = os.path.join(root,file)
    try:
        img = imread(filename).astype('float32')
    except:
        print(filename)