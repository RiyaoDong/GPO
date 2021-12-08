import os
import shutil

f = open("data/tiny_imagenet/val/val_annotations.txt", "r")
text_lines = f.readlines()
for line in text_lines:
    lines = line.split("\t")
    print(lines[0], lines[1])
    if not os.path.exists("data/tiny_imagenet/valset/"+lines[1]):
        os.mkdir("data/tiny_imagenet/valset/"+lines[1])
    shutil.copy("data/tiny_imagenet/val/images/"+lines[0], "data/tiny_imagenet/valset/"+lines[1]+"/"+lines[0])
f.close()
