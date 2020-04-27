import sys
from PIL import Image

if __name__ == "__main__":
	basewidth = 800
	img = Image.open(sys.argv[1])
	wpercent = (basewidth/float(img.size[0]))
	hsize = int((float(img.size[1])*float(wpercent)))
	img = img.resize((basewidth,hsize), Image.ANTIALIAS)
	img.save(sys.argv[1])
