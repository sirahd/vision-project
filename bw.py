#!/usr/bin/env python

import Image
import os, sys

def resizeImage(infile, output_dir="", size=[256,256]):
     outfile = os.path.splitext(infile)[0]
     extension = os.path.splitext(infile)[1]

     if (cmp(extension, ".jpg")):
        return

     if infile != outfile:
        infile = "resized/" + infile
        try :
            im = Image.open(infile)
            im = im.convert('1')
            im.save(output_dir+"/"+outfile+extension,"JPEG")
        except IOError as e:
            print e.errno
            print e

if __name__=="__main__":
    output_dir = "bw"
    dir = os.getcwd() + "/resized"

    if not os.path.exists(os.path.join(os.getcwd(),output_dir)):
        os.mkdir(output_dir)

    for file in os.listdir(dir):
        resizeImage(file, output_dir)
