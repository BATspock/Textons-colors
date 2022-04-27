from PIL import Image
import os
images = os.listdir("/home/adkishor/projects/Textons-colors/images/")

for imgs in images:

	og_image = Image.open("/home/adkishor/projects/Textons-colors/images/"+str(imgs))

	for angle in range(0, 360, 90):
	    im = og_image.rotate(angle)
	    im.save("/home/adkishor/projects/Textons-colors/dataset/"+str(imgs)+"_"+str(angle)+".png")
