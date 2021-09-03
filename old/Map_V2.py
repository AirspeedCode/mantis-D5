#!/usr/bin/python3

#from PyQt5 import QtWidgets
#from pyqtgraph import PlotWidget, plot

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg



import cv2
import datetime
from math import radians, cos, sin, asin, sqrt
import numpy as np
import os
import pandas as pd

#from pyqtgraph.Qt import QtGui, QtCore
#from PyQt5 import QtGui, QtCore
#import pyqtgraph as pg




import sys  # We need sys so that we can pass argv to QApplication

def DJI_to_UTC(time_str):
    year = int(time_str[:4])
    month = int(time_str[5:7])
    day = int(time_str[8:10])
    hour = int(time_str[11:13])
    minute = int(time_str[14:16])
    seconds = float(time_str[17:])
    microsecond = int(1e6 *  (seconds % 1))
    second = int(seconds - (seconds % 1))
    
    #print(year, month, day, hour, minute, seconds, second, microsecond)
    time_var = datetime.datetime(year, month, day, hour, minute, second, microsecond)
    
    epoch = datetime.datetime.utcfromtimestamp(0)
    utc_time = (time_var - epoch).total_seconds()
    return utc_time

def LoadFlightLog(filename):
    #Load the data as a pandas dataframe
    df = pd.read_csv('DJIFlightRecord_2019-12-17_[15-43-22]-TxtLogToCsv.csv', sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

    # remove the unnecessary columns from the dataframe
    header_list = list(df[:0])
    required_columns = ['CUSTOM.updateTime', 'OSD.flyTime [s]',  'OSD.latitude', 'OSD.longitude', 'OSD.height [m]', 'OSD.groundOrSky']
    for header in header_list:
        if(not(header in required_columns)):
            del df[header]

    # Add a UTC timestamp to the dataframe
    UTC_list = []
    for DJI_stamp in list(df['CUSTOM.updateTime']):
        #print(DJI_stamp)
        UTC_seconds = DJI_to_UTC(DJI_stamp)
        UTC_list.append(UTC_seconds)

    df['UTC_Seconds'] = UTC_list

    #overwrite the datatypes of dataframe
    data_types = df.astype({
                'CUSTOM.updateTime': 'string',
                'OSD.latitude': 'float64',
                'OSD.longitude': 'float64',
                'OSD.height [m]': 'float64',
                'OSD.groundOrSky': 'string',
                'OSD.flyTime [s]': 'float64',
                'UTC_Seconds': 'float64'
                }).dtypes

    df = df.astype(data_types)
    return df

def GetDroneStatus(df, UTC):
    drone_status = df[df['UTC_Seconds'] <= UTC].iloc[-1]
    lat = drone_status['OSD.latitude']
    lon = drone_status['OSD.longitude']
    height = drone_status['OSD.height [m]']
    groundOrSky = drone_status['OSD.groundOrSky']
    Flytime = drone_status['OSD.flyTime [s]']
    Timestamp = drone_status['UTC_Seconds']
    return Timestamp, Flytime, lat, lon, height, groundOrSky

def GetStartUTC(df):
    return df.iloc[0]['UTC_Seconds']

def GetEndUTC(df):
    return df.iloc[-1]['UTC_Seconds']

def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371000 # Radius of earth in meters. 
        return c * r    

class map_point(object):
    def __init__(self, lon, lat):
        print("Lat, Lon:", lat, lon)
        
        self.X0_lon = -0.472412109375 
        self.Y0_lat = 51.984880139916626
        
        self.X = 0    # X Position on the image in (m) relative to bottom left corner (0,0)
        self.Y = 0    # Y Position on the image in (m) relative to bottom left corner (0,0)
        self.degreesToXYPos(lat,lon)
        
    def degreesToXYPos(self, lat, lon):
        print("Lat, Lon:", lat, lon)
        
        print(self.X0_lon, self.Y0_lat, lon, self.Y0_lat )
        
        self.X = haversine( lon1=self.X0_lon, lat1=self.Y0_lat, lon2=lon, lat2=self.Y0_lat )
        self.Y = haversine( lon1=self.X0_lon, lat1=self.Y0_lat, lon2=self.X0_lon, lat2=lat )
        

class MANTIS_player(object):
    def __init__(self):


        # pyqtgraph stuff
        pg.setConfigOptions(antialias=True)

        ## Switch to using white background and black foreground
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        self.app = QtGui.QApplication(sys.argv)
        self.win = pg.GraphicsWindow(title='MANTIS Field Trial Player')    
        self.win.ci.layout.setContentsMargins(50, 50, 50, 50) # margins around the edge of the window
        self.win.setWindowTitle('MANTIS Field Trial Player')
        self.win.setGeometry(50, 50, 1200, 800)
        self.map_plot = self.win.addPlot()
        self.map_plot.setAspectLocked()

        #Load the drone flight log
        filename = 'DJIFlightRecord_2019-12-17_[15-43-22]-TxtLogToCsv.csv'
        self.FlightData = LoadFlightLog(filename) # returns a pandas dataframe

        

        # Get drone initial conditions
        self.startUTC = GetStartUTC(self.FlightData)
        self.simulation_UTC = self.startUTC
        #self.simulation_elapsed_time = 0.0
        self.drone_timestamp, self.drone_flytime, self.drone_lat, self.drone_lon, self.drone_height, self.drone_groundOrSky = GetDroneStatus(df = self.FlightData, UTC = self.simulation_UTC)


        # Intantiate a mote and drone object
        self.drone = map_point(lon = self.drone_lon, lat = self.drone_lat)
        self.mote = map_point(lon = -0.455210, lat = 51.999779)


        # Load the satallite background image, store in 'map_imp' numpy array
        filename  = './composite_images/satellite.png'
        img_array = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        map_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB) # restore correct RGB coding
        map_img = np.rot90(map_img,3)
        img = pg.ImageItem(map_img)

        map_x = 0
        map_y = 0
        map_width = 2256
        map_height = 3009

        bounds = pg.QtCore.QRect(map_x, map_y, map_width, map_height)
        img.setRect(bounds)
        self.map_plot.addItem(img)

        #add range circles

        for circle in range (7):
            increment = 50
            radius = circle * increment
            circle = pg.QtGui.QGraphicsEllipseItem(self.mote.X-radius, self.mote.Y-radius, radius * 2, radius * 2)
            circle.setPen(pg.mkPen(color=(200,200,200), style=QtCore.Qt.SolidLine, width=0.75))
            self.map_plot.addItem(circle)

            range_str = '{} m'.format(int(radius))
            range_circle_label = pg.TextItem(text=range_str, color=(200,200,200), fill=None)
            range_circle_label.setFont(pg.QtGui.QFont('Arial', 10))

            range_circle_label.setPos(self.mote.X , self.mote.Y + radius )

            if(radius > 0):
                self.map_plot.addItem(range_circle_label)



        # Add the plots to the window
        self.traces = dict()

        self.map_plot.addLegend(brush=(200,200,200))
        #self.map_plot.addLegend(brush=pg.mkBrush(255, 255, 255, 120))
        class waveplot:
            def __init__(self, plotType, plotItem):
                self.plotType = plotType
                self.plotItem =  plotItem
                self.plotItem.setYRange(1300, 2000, padding=0.0) 
                self.plotItem.setXRange(600, 1800, padding=0.0) 

                if self.plotType == 'DroneTrackType':
                    # Add the plot pane to the window    
                    self.trace = self.plotItem.plot( 
                        pen=(0,0,200), 
                        symbolBrush=(200,0,0), 
                        symbolPen='w', 
                        symbolSize=14, 
                        symbol='o', 
                        name="DRONE")

                elif self.plotType == 'MoteTrackType':
                    # Add the plot pane to the window
                    self.trace = self.plotItem.plot(
                        pen=(0,0,0), 
                        symbolBrush=(0,0,255), 
                        symbolPen='w', 
                        symbolSize=14, 
                        symbol='h', 
                        name="MOTE")


        self.DroneTrack = waveplot(plotType = 'DroneTrackType', plotItem = self.map_plot)
        self.MoteTrack = waveplot(plotType = 'MoteTrackType', plotItem = self.map_plot)


        #Draw the mote
        mote_x_pts = [self.mote.X]
        mote_y_pts = [self.mote.Y]
        self.MoteTrack.trace.setData(mote_x_pts, mote_y_pts)

       


        
    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self):

        #self.centre_x += (10/10)
        #print(self.centre_x, self.centre_y, self.r)

        time_step = 0.1 #seconds
        self.simulation_UTC += time_step
        self.drone_timestamp, self.drone_flytime, self.drone_lat, self.drone_lon, self.drone_height, self.drone_groundOrSky = GetDroneStatus(df = self.FlightData, UTC = self.simulation_UTC)
        self.drone.degreesToXYPos(self.drone_lat, self.drone_lon)


        drone_x_pts = [self.drone.X]
        drone_y_pts = [self.drone.Y]
        self.DroneTrack.trace.setData(drone_x_pts, drone_y_pts)


    

    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(100)
        self.start()






if __name__ == '__main__':
    map_app = MANTIS_player()
    map_app.animation()

