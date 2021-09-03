from PyQt5 import QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys  # We need sys so that we can pass argv to QApplication
import numpy as np
import os
#from PIL import Image
import cv2

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.graphWidget = pg.PlotWidget()
        self.setCentralWidget(self.graphWidget)

        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]

        self.graphWidget.setBackground('w')
        #self.graphWidget.plot(hour, temperature)

        # Load the image
        filename  = './composite_images/satellite.png'
        #img = Image.open(filename)

        #out = img.rotate(270)
        
        #image = np.array(img).T

        #img_array = cv2.imread(filename, cv2.IMREAD_GRAYSCALE).T
        img_array = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        RGB_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # restore correct RGB coding

        RGB_img = cv2.flip(RGB_img,0)
        RGB_img = np.rot90(RGB_img,3)




        #print(image)







        self.graphWidget = pg.image(RGB_img)


def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

