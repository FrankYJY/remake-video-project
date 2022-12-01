import os

def runJava(parentDict,folder):
  os.system("javac RGB2PNG.java")
  os.system("java RGB2PNG "+ parentDict + ' ' + folder)
  
def getPath():
  pngPathFiles = open("pngPaths.txt", "r")
  frameNumber =0 
  pngPaths =[]
  for line in pngPathFiles:
    #print(line[:-1]) # del /n
    pngPaths.append(line[:-1])  # del /n
    frameNumber += 1
  pngPathFiles.close()
  return frameNumber,pngPaths
  
  
#your path
parentDict="" #可以不填 如果该py文件和folder同级
folder = "SAL_490_270_437"

runJava(parentDict,folder)
frameNumber,pngPaths= getPath()

#The Result Please use me !!!
print(frameNumber)
print(pngPaths)