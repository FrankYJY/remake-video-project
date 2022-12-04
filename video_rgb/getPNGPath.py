import os

def runJavaRGB2PNG(parentDict,folder):
  # parameters eg: 'C:/Users/14048/Desktop/multimedia/project/video_rgb/', 'SAL_490_270_437'
  if not os.path.exists("RGB2PNG.class"):
    os.system("javac RGB2PNG.java")
  os.system("java RGB2PNG "+ parentDict + ' ' + folder)
  
def getPNGPathes():
  # if not os.path.exists("pngPaths.txt"):
  #   open("pngPaths.txt", "w")
  pngPathFiles = open("pngPaths.txt", "r")
  frameNumber = 0 
  pngPaths =[]
  for line in pngPathFiles:
    #print(line[:-1]) # del /n
    pngPaths.append(line[:-1])  # del /n
    frameNumber += 1
  pngPathFiles.close()
  return frameNumber,pngPaths
  
if __name__ == "__main__":
  #your path
  parentDict="C:/Users/14048/Desktop/multimedia/project/video_rgb/" #可以不填 如果该py文件和folder同级
  folder = "motiontestimg_490_270_003"

  runJavaRGB2PNG(parentDict,folder)
  frameNumber,pngPaths= getPNGPathes()

  #The Result Please use me !!!
  print(frameNumber)
  print(pngPaths)