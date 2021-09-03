#!/usr/bin/python3

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import cv2
import datetime
from math import radians, cos, sin, asin, sqrt
import numpy as np
import os
import pandas as pd
import math
import sys  # We need sys so that we can pass argv to QApplication


class pens(object):
    def __init__(self, color = 'None', type = 'None', width = 'None'):
         # Colors
        
        self.black      = pg.QtGui.QColor(0, 0, 0)
        self.blue       = pg.QtGui.QColor(0, 0, 255)
        self.green      = pg.QtGui.QColor(0, 255, 0)
        self.cyan       = pg.QtGui.QColor(0, 255, 255)
        self.red        = pg.QtGui.QColor(255, 0, 0)
        self.magenta    = pg.QtGui.QColor(255, 0, 255)
        self.yellow     = pg.QtGui.QColor(255, 255, 0)
        self.white      = pg.QtGui.QColor(255, 255, 255)
        self.darkRed    = pg.QtGui.QColor(102, 0, 0)
        self.darkBlue   = pg.QtGui.QColor(0, 0, 102)
        self.grey       = pg.QtGui.QColor(128, 128, 128)        
        self.lightGrey  = pg.QtGui.QColor(192, 192, 192)
        self.darkGrey   = pg.QtGui.QColor(64, 64, 64)

        # Dashed line pens
        self.dashedBluePen = pg.mkPen(color=(self.darkBlue), style=QtCore.Qt.DotLine, width=2)
        self.dashedRedPen = pg.mkPen(color=(self.red), style=QtCore.Qt.DotLine, width=2)
        self.dashedGreenPen = pg.mkPen(color=(self.green), style=QtCore.Qt.DotLine, width=2)
        self.dashedBlackPen = pg.mkPen(color=(self.black), style=QtCore.Qt.DotLine, width=2)

        # Solid Pens
        self.solidBluePen = pg.mkPen(color=(self.darkBlue), style=QtCore.Qt.SolidLine, width=2)
        self.solidRedPen = pg.mkPen(color=(self.red), style=QtCore.Qt.SolidLine, width=2)
        self.solidGreenPen = pg.mkPen(color=(self.green), style=QtCore.Qt.SolidLine, width=2)
        self.solidBlackPen = pg.mkPen(color=(self.black), style=QtCore.Qt.SolidLine, width=2)
        self.solidGreyPen = pg.mkPen(color=(self.darkGrey), style=QtCore.Qt.SolidLine, width=2)

        # Fine pens
        self.fineGreenPen = pg.mkPen(color=(self.green), style=QtCore.Qt.SolidLine, width=1.0)
        self.fineYellowPen = pg.mkPen(color=(self.yellow), style=QtCore.Qt.SolidLine, width=1.0)

        # Construction/Grid Line pens
        self.lightGridPen = pg.mkPen(color=(self.grey), style=QtCore.Qt.SolidLine, width=0.2)
        self.heavyGridPen = pg.mkPen(color=(self.darkGrey), style=QtCore.Qt.SolidLine, width=0.75)

        # returns a user defined 'Custom' pen
        if   color == 'black':      penColor = self.black
        elif color == 'blue':       penColor = self.blue
        elif color == 'green':      penColor = self.green
        elif color == 'cyan':       penColor = self.cyan
        elif color == 'red':        penColor = self.red
        elif color == 'magenta':    penColor = self.magenta
        elif color == 'yellow':     penColor = self.yellow
        elif color == 'white':      penColor = self.white
        elif color == 'darkRed':    penColor = self.darkRed
        elif color == 'darkBlue':   penColor = self.darkBlue
        elif color == 'grey':       penColor = self.grey
        elif color == 'lightGrey':  penColor = self.lightGrey
        elif color == 'darkGrey':   penColor = self.darkGrey
        else:
            penColor = self.white
            print("pens class: Invalid pen colour")
        
        if type == 'dashed':    penType = QtCore.Qt.DotLine
        elif type == 'solid':   penType = QtCore.Qt.SolidLine
        else:
            penType = QtCore.Qt.SolidLine
            print("pens class: Invalid pen style")
        
        if   width == 'thick':      penWidth = 2.0
        elif width == 'very thick': penWidth = 5.0
        elif width == 'medium':     penWidth = 1.5
        elif width == 'fine':       penWidth = 1.0
        elif width == 'feint':      penWidth = 0.75
        elif width == 'very feint': penWidth = 0.2
        else:
            penWidth = 1.0
            print("pens class: Invalid pen width")

        self.customPen = pg.mkPen(color = penColor, style = penType, width = penWidth)

def LoadLitchiFlightLog(filename):
    #Load the data as a pandas dataframe
    df = pd.read_csv(filename, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

    # remove the unnecessary columns from the dataframe
    header_list = list(df[:0])
    required_columns = ['timestamp', 'time(millisecond)',  'latitude', 'longitude', 'altitude(m)', 'isflying']
    for header in header_list:
        if(not(header in required_columns)):
            del df[header]

    #overwrite the datatypes of dataframe
    data_types = df.astype({
                'timestamp': 'float64',
                'time(millisecond)': 'float64',
                'latitude': 'float64',
                'longitude': 'float64',
                'altitude(m)': 'float64',
                'isflying': 'int',
                }).dtypes
    df = df.astype(data_types)

    print(df.head())

    return df

def GetEndUTC(df):
    return df.iloc[-1]['timestamp']

def GetStartUTC(df):
    startUTC = df.iloc[0]['timestamp']
    print(">>>>>>>> START UTC:", startUTC)
    return startUTC

def GetDroneStatus(df, UTC):
    #print(">>>>>>>> UTC:", UTC)
    drone_status = df[df['timestamp'] <= UTC].iloc[-1]
    lat = drone_status['latitude']
    lon = drone_status['longitude']
    height = drone_status['altitude(m)']
    groundOrSky = drone_status['isflying']
    Flytime = drone_status['time(millisecond)']
    Timestamp = drone_status['timestamp']
    return Timestamp, Flytime, lat, lon, height, groundOrSky

def LoadCNNLog(fname):
    #Load the data as a pandas dataframe
    df = pd.read_csv(fname, sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

    #overwrite the datatypes of dataframe
    data_types = df.astype({
                'UTC (ms)': 'float64',
                'CNN Confidence': 'float64',
                'CNN Bearing': 'float64',
                'Confidence (Filtered)': 'float64',
                'Bearing (Filtered)': 'float64',
                'Drone Detected': 'float64',
                }).dtypes

    df = df.astype(data_types)
    return df

def GetCNNStatus(df, UTC):
    #print(">>>>>>>> UTC:", UTC)
    drone_status = df[df['UTC (ms)'] <= UTC].iloc[-1]
    TIME = drone_status['UTC (ms)']
    BRG = drone_status['Bearing (Filtered)']
    CONF = drone_status['Confidence (Filtered)']
    ISDRONE = drone_status['Drone Detected']
    return TIME, BRG, CONF, ISDRONE

def GetCNNLogStartUTC(df):
    startUTC = df.iloc[0]['UTC (ms)']
    print(">>>>>>>> CNN LOG START UTC:", startUTC)
    return startUTC


def GetCNNLogEndUTC(df):
    endUTC = df.iloc[-1]['UTC (ms)']
    print(">>>>>>>> CNN LOG END UTC:", endUTC)
    return endUTC

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

''' A class for handling points on the map display
'''
class map_point(object):
    def __init__(self, lon, lat):
        #print("Lat, Lon:", lat, lon)
        
        self.X0_lon = -0.472412109375 
        self.Y0_lat = 51.984880139916626
        
        self.X = 0    # X Position on the image in (m) relative to bottom left corner (0,0)
        self.Y = 0    # Y Position on the image in (m) relative to bottom left corner (0,0)
        self.degreesToXYPos(lat,lon)
        
    def degreesToXYPos(self, lat, lon):
        #print("Lat, Lon:", lat, lon)
        
        #print(self.X0_lon, self.Y0_lat, lon, self.Y0_lat )
        
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

        #select which mission to replay
        #mission = 'P1D4T8F1'
        #mission = 'P1D4T8F2a'
        #mission = 'P1D4T8F2b'
        mission = 'P1D4T8F3'
        #mission = 'P1D4T8F4'

        if mission == 'P1D4T8F1':        
            FlightLogFname = './DJI_Flight_Logs/2021-08-11_17-59-07_v2.csv'
            MoteLogFname = './CNN_Logs/1628701000496_prediction_log.csv'
            self.mote = map_point(lon = -0.45495, lat = 51.99967)
            self.mission_str = "P1D4 TRIAL 8, FLIGHT 1"

        elif mission == 'P1D4T8F2a':
            FlightLogFname = './DJI_Flight_Logs/2021-08-11_18-18-09_v2.csv'
            MoteLogFname = './CNN_Logs/1628702203042_prediction_log.csv'
            self.mote = map_point(lon = -0.45495, lat = 51.99967)
            self.mission_str = "P1D4 TRIAL 8, FLIGHT 2a"
        
        elif mission == 'P1D4T8F2b':
            FlightLogFname = './DJI_Flight_Logs/2021-08-11_18-28-06_v2.csv'
            MoteLogFname = './CNN_Logs/1628703640950_prediction_log.csv'
            self.mote = map_point(lon = -0.45495, lat = 51.99967)
            self.mission_str = "P1D4 TRIAL 8, FLIGHT 2b"

        elif mission == 'P1D4T8F3':
            FlightLogFname = './DJI_Flight_Logs/2021-08-11_19-05-51_v2.csv'
            MoteLogFname = './CNN_Logs/1628705107648_prediction_log.csv'
            self.mote = map_point(lon = -0.45495, lat = 51.99967)
            self.mission_str = "P1D4 TRIAL 8, FLIGHT 3"

        elif mission == 'P1D4T8F4':
            FlightLogFname = './DJI_Flight_Logs/2021-08-11_19-05-51_v2.csv'
            MoteLogFname = './CNN_Logs/1628707327839_prediction_log.csv'
            self.mote = map_point(lon = -0.45495, lat = 51.99967)
            self.mission_str = "P1D4 TRIAL 8, FLIGHT 4"


        #Load the drone flight log
        self.FlightData = LoadLitchiFlightLog(FlightLogFname) # returns a pandas dataframe

        #Load the CNN Log file data        
        self.CNNData = LoadCNNLog(MoteLogFname)
        self.CNNDataStartUTC = GetCNNLogStartUTC(self.CNNData)
        self.CNNDataEndUTC = GetCNNLogEndUTC(self.CNNData)

        self.CNN_UTC = 0.0 
        self.CNN_BRG = 0.0
        self.CNN_CONF = 0.0
        self.CNN_ISDRONE = 0.0

        # Get drone initial conditions
        self.FlightDataStartUTC = GetStartUTC(self.FlightData)
        self.FlightDataEndUTC = GetEndUTC(self.FlightData)


        if self.FlightDataStartUTC > self.CNNDataEndUTC:
            print(">>> ERROR: Flight Log starts after Mote Log Ends")
            print("Flight Log File: {}".format(FlightLogFname))
            print("Mote Log File: {}".format(MoteLogFname))
            print("Flight log: {} to {}".format(int(self.FlightDataStartUTC),int(self.FlightDataEndUTC)))
            print("Mote log: {} to {}".format(int(self.CNNDataStartUTC),int(self.CNNDataEndUTC)))
            exit()

        if self.CNNDataStartUTC > self.FlightDataEndUTC:
            print("Flight Log File: {}".format(FlightLogFname))
            print("Mote Log File: {}".format(MoteLogFname))
            print(">>> ERROR: Mote Log starts after Flight Log Ends")
            print("Flight log: {} to {}".format(int(self.FlightDataStartUTC),int(self.FlightDataEndUTC)))
            print("Mote log: {} to {}".format(int(self.CNNDataStartUTC),int(self.CNNDataEndUTC)))
            exit()
        

        self.simulation_UTC = min(self.FlightDataStartUTC, self.CNNDataStartUTC)      
        self.simulation_Start_UTC = min(self.FlightDataStartUTC, self.CNNDataStartUTC)
        self.simulation_End_UTC = max(self.FlightDataEndUTC, self.CNNDataEndUTC)
        self.simulation_length = int(self.simulation_End_UTC - self.simulation_UTC)

        print("Simulation start UTC:", self.simulation_UTC)
        print("Simulation end UTC:", self.simulation_End_UTC)

        self.offset_time = 500 # (in ms), +VE means mote UTC lags drone, -VE means drone lags mote

        #self.simulation_elapsed_time = 0.0
        self.drone_timestamp, self.drone_flytime, self.drone_lat, self.drone_lon, self.drone_height, self.drone_groundOrSky = GetDroneStatus(df = self.FlightData, UTC = self.FlightDataStartUTC)
        
        # Intantiate a mote and drone object
        self.drone = map_point(lon = self.drone_lon, lat = self.drone_lat)
        

        # Load the satallite background image, store in 'map_img' numpy array
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
                        symbolSize=16, 
                        symbol='o', 
                        name="DRONE")

                elif self.plotType == 'MoteTrackType':
                    # Add the plot pane to the window
                    self.trace = self.plotItem.plot(
                        pen=(0,0,0), 
                        symbolBrush=(0,0,255), 
                        symbolPen='w', 
                        symbolSize=16, 
                        symbol='h', 
                        name="MOTE")


        self.DroneTrack = waveplot(plotType = 'DroneTrackType', plotItem = self.map_plot)
        self.MoteTrack = waveplot(plotType = 'MoteTrackType', plotItem = self.map_plot)

        #Draw the mote
        mote_x_pts = [self.mote.X]
        mote_y_pts = [self.mote.Y]
        self.MoteTrack.trace.setData(mote_x_pts, mote_y_pts)

        # Simulation Time Step (ms)
        self.Sim_Time_Step = 50
        self.Playback_Speed_Factor = 4 # Fast forward speed (1 to 10), 1 = realtime, 10 = 10 times faster

        #self.MarkerPen = pens(color = 'red', type = 'solid', width = 'very thick').customPen
        self.MarkerPen = pens(color = 'yellow', type = 'solid', width = 'very thick').customPen
        self.StaleMarkerPen = pens(color = 'red', type = 'dashed', width = 'very thick').customPen

        #self.PolarMarker = pg.QtGui.QGraphicsLineItem(0,0, -np.sin(brg_rad), np.cos(brg_rad))
        self.PolarMarker = pg.QtGui.QGraphicsLineItem()

        self.PolarMarker.setLine(self.mote.X,self.mote.Y, self.mote.X +  (300 * np.sin(radians(0))) , self.mote.Y + (300 * np.cos(radians(0))) )

        #self.PolarMarker.setLine(self.mote.X,self.mote.Y, self.mote.X + self.sin_theta[int(0)], self.mote.Y + self.cos_theta[int(0)])

        self.PolarMarker.setPen(self.MarkerPen)
        self.MoteTrack.plotItem.addItem(self.PolarMarker)

        # Drone Status Label
        self.DroneStatuses = ['GROUND', 'FLYING']
        self.DroneStatText = 'DRONE STATUS: {}'.format(self.DroneStatuses[0])
        self.DroneStatLabel = pg.TextItem(text=self.DroneStatText, anchor=(0.0,0.0), color = pg.QtGui.QColor(255, 255, 255), fill=pg.mkBrush(color=pg.QtGui.QColor(0, 0, 0)))
        self.DroneStatLabel.setFont(pg.QtGui.QFont('Arial', 16))
        self.DroneStatLabel.setPos( 1700, 1980 )
        self.map_plot.addItem(self.DroneStatLabel)

          
    def start(self):
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self):

        self.simulation_UTC += self.Sim_Time_Step

        DroneStatusStr = self.DroneStatuses[0]
        FlightLogStr = "NO DATA"
        CNNLogStr = "NO DATA"
      
        if (self.simulation_UTC >= self.FlightDataStartUTC and self.simulation_UTC <= self.FlightDataEndUTC):
            self.drone_timestamp, self.drone_flytime, self.drone_lat, self.drone_lon, self.drone_height, self.drone_groundOrSky = GetDroneStatus(df = self.FlightData, UTC = self.simulation_UTC)
            DroneStatusStr = self.DroneStatuses[int(self.drone_groundOrSky)]
            FlightLogStr = "RUNNING"
        else:
            #print("Flight Log Data Out of range")    
            FlightLogStr = "NO DATA"
        
        mote_time = self.simulation_UTC + self.offset_time

        if (mote_time >= self.CNNDataStartUTC and mote_time <= self.CNNDataEndUTC):
            self.CNN_UTC, self.CNN_BRG, self.CNN_CONF, self.CNN_ISDRONE = GetCNNStatus(df = self.CNNData, UTC = mote_time)
            #print(self.CNN_UTC, self.CNN_BRG, self.CNN_CONF, self.CNN_ISDRONE)
            CNNLogStr = "RUNNING"


            # Update the Mote Polar Marker
            #self.PolarMarker.setLine(0,0, self.sin_theta[int(kf_theta)], self.cos_theta[int(kf_theta)])

            if self.CNN_ISDRONE == 1.0:
                
                if not self.PolarMarker in self.MoteTrack.plotItem.items:
                    self.MoteTrack.plotItem.addItem(self.PolarMarker)
                self.PolarMarker.setLine(self.mote.X,self.mote.Y, self.mote.X +  (300 * np.sin(radians(self.CNN_BRG))) , self.mote.Y + (300 * np.cos(radians(self.CNN_BRG))) )
                    
            else:
                if self.PolarMarker in self.MoteTrack.plotItem.items:
                    self.MoteTrack.plotItem.removeItem(self.PolarMarker)
                
        else:
            #print("CNN Data Out of range")
            CNNLogStr = "NO DATA"


        self.drone.degreesToXYPos(self.drone_lat, self.drone_lon)
        drone_x_pts = [self.drone.X]
        drone_y_pts = [self.drone.Y]
        self.DroneTrack.trace.setData(drone_x_pts, drone_y_pts)


        # Update the drone status box
        progress_pc = (100 * float(self.CNN_UTC - self.simulation_Start_UTC)/float(self.simulation_length))
        self.DroneStatText = 'MISSION: {} \nUTC: {}\t({:4.1f}% )\nPLAYBACK SPEED: {}X \nDRONE STATUS: {} \nFLIGHT LOG: {} \nMOTE LOG: {} '.format(self.mission_str, int(self.CNN_UTC), progress_pc, int(self.Playback_Speed_Factor), DroneStatusStr, FlightLogStr, CNNLogStr)
        self.DroneStatLabel.setText(self.DroneStatText)


    def animation(self):
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start( int(self.Sim_Time_Step / self.Playback_Speed_Factor) )
        self.start()


if __name__ == '__main__':
    map_app = MANTIS_player()
    map_app.animation()

