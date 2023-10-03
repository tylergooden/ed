import sys
import cv2
import numpy as np
import noise

from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QMenuBar, QMenu, QFileDialog, QVBoxLayout, QComboBox, QWidget, QSlider 
from PySide6.QtGui import QPixmap, QImage, QColor, QAction
from PySide6.QtCore import Qt, Signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg 


class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Default dimension for square images
        self.dim = 512

        # Share Numpy arrays for efficiency: xcoord, xycoords
        self.npImgCont = np.ones((self.dim, self.dim), dtype=np.float32)
        # Linearly divide range between 0 and 1 to make np array.
        self.xcoords = np.linspace(0, 1, self.dim)
        # Make similar structure in 2D similar to UV coords. 
        x, y = np.meshgrid(self.xcoords, self.xcoords)
        self.xycoords = np.dstack((x, y))
        

        # Make an empty list for undo stack
        self.history = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('ED')
        self.resize(self.dim, self.dim) 

        # Image display label
        self.imageLabel = QLabel(self)
        self.imageLabel.setMinimumSize(1, 1)  # Allow downsizing
        self.setCentralWidget(self.imageLabel)

        # Menu bar
        menubar = self.menuBar()
        fileMenu = QMenu('File', self)
        editMenu = QMenu('Edit', self)
        toolsMenu = QMenu('Tools', self)
        menubar.addMenu(fileMenu)
        menubar.addMenu(editMenu)
        menubar.addMenu(toolsMenu)

        openAction = QAction('Open', self)
        openAction.triggered.connect(self.openImage)
        saveAction = QAction('Save', self)
        saveAction.triggered.connect(self.saveImage)
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)

        grayBasicAction = QAction('Convert to grayscale basic', self)
        grayBasicAction.triggered.connect(self.convertToGrayBasic)
        grayNumpyAction = QAction('Convert to grayscale NumPy', self)  
        grayNumpyAction.triggered.connect(self.convertToGrayNumpy)    
        grayOpenCVAction = QAction('Convert to grayscale OpenCV', self)
        grayOpenCVAction.triggered.connect(self.convertToGrayOpenCV)
        edgeDetectionAction = QAction('Run edge detection', self)
        edgeDetectionAction.triggered.connect(self.runEdgeDetection)
        undoAction = QAction('Undo', self)
        undoAction.setShortcut('Ctrl+Z')  
        undoAction.triggered.connect(self.undo)
        editMenu.addAction(grayBasicAction)
        editMenu.addAction(grayNumpyAction) 
        editMenu.addAction(grayOpenCVAction)
        editMenu.addAction(edgeDetectionAction)
        editMenu.addAction(undoAction)

        showFunctionWindowAction = QAction("Show Function Window", self)
        showFunctionWindowAction.triggered.connect(self.show_function_window)
        perlinNoiseAction = QAction('Generate Perlin noise', self)
        perlinNoiseAction.triggered.connect(self.generatePerlinNoise)
        toolsMenu.addAction(showFunctionWindowAction)
        toolsMenu.addAction(perlinNoiseAction)

        self.show()

    def resizeEvent(self, event):
        if hasattr(self, 'pixmap'):
            self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))


    def pushToHistory(self):
        if hasattr(self, 'pixmap'):
            self.history.append(self.pixmap.copy())

# File

    def openImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open image', '.', 'Image files (*.png *.jpg *.bmp)')
        if fname:
            self.pushToHistory()
            self.pixmap = QPixmap(fname)
            self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))

    def saveImage(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Save image', '.', 'Image files (*.png *.jpg *.bmp)')
        if fname:
            self.pixmap.save(fname)


# Edit

    def convertToGrayBasic(self):
        self.pushToHistory()
        image = self.pixmap.toImage()
        for x in range(image.width()):
            for y in range(image.height()):
                c = image.pixelColor(x, y)
                gray = int(0.3 * c.red() + 0.59 * c.green() + 0.11 * c.blue())
                image.setPixelColor(x, y, QColor(gray, gray, gray))
        self.pixmap = QPixmap.fromImage(image)
        self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))

    def convertToGrayNumpy(self):
        self.pushToHistory()
        arr = self.pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        img_data = np.frombuffer(arr.constBits(), dtype=np.uint8).reshape((arr.height(), arr.width(), 3))
        gray = np.dot(img_data, [0.3, 0.59, 0.11])
        self.pixmap = QPixmap.fromImage(QImage(gray.astype(np.uint8).data, gray.shape[1], gray.shape[0], QImage.Format_Grayscale8))
        self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))

    def convertToGrayOpenCV(self):
        self.pushToHistory()
        arr = self.pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        img_data = np.frombuffer(arr.constBits(), dtype=np.uint8).reshape((arr.height(), arr.width(), 3))
        img = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        self.pixmap = QPixmap.fromImage(QImage(img.data, img.shape[1], img.shape[0], QImage.Format_Grayscale8))
        self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))

    def runEdgeDetection(self):
        self.pushToHistory()
        arr = self.pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
        img_data = np.frombuffer(arr.constBits(), dtype=np.uint8).reshape((arr.height(), arr.width(), 3))
        gray = cv2.cvtColor(img_data, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        self.pixmap = QPixmap.fromImage(QImage(edges.data, edges.shape[1], edges.shape[0], QImage.Format_Grayscale8))
        self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))


# Tools

    def show_function_window(self):
        self.functionWindow = FunctionWindow(self.npImgCont, self.dim, self.xcoords, self.xycoords)
        self.functionWindow.functionChanged.connect(self.on_function_changed)
        self.functionWindow.show()

    # Signal handler setup by slot -- handle image change when function changed
    def on_function_changed(self, func_type, param):

        # Update pixmap by converting range, to QImage, and setting imageLabel
        fImageData = (self.npImgCont * 255).astype(np.uint8)
        self.pixmap = QPixmap.fromImage(QImage(fImageData.astype(np.uint8).data, fImageData.shape[1], fImageData.shape[0], QImage.Format_Grayscale8))
        self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))


    def generatePerlinNoise(self):
        self.pushToHistory()
        img = np.zeros((self.dim, self.dim, 1), dtype=np.uint8)

        shape = (self.dim, self.dim)
        scale = .5
        octaves = 6
        persistence = 0.5
        lacunarity = 2.0
        seed = np.random.randint(0,100)
        
        world = np.zeros(shape)
        
        # TODO:  continue to modify to use xycoords soon.
        # Make coordinate grid of [0,1] by [0,1].
        world_x, world_y = np.meshgrid(self.xcoords, self.xcoords)
        
        # Apply "Improved Perlin" noise.
        world = np.vectorize(noise.pnoise2)(world_x/scale, world_y/scale,
                                octaves=octaves, persistence=persistence, lacunarity=lacunarity,
                                repeatx=self.dim, repeaty=self.dim,
                                base=seed)
        
        #  Re-range image values. 
        img = np.floor((world + 0.5) * 255).astype(np.uint8)
        
        self.pixmap = QPixmap.fromImage(QImage(img.data, img.shape[1], img.shape[0], QImage.Format_Grayscale8))
        self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))


    def undo(self):
        if self.history:
            tmpPixmap = self.pixmap.copy()
            self.pixmap = self.history.pop()
            self.imageLabel.setPixmap(self.pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))
            self.history.append(tmpPixmap)



#################
# Function Window
#################

class FunctionWindow(QMainWindow):

    # Signal to let parent window (main in this case) know there is an update
    functionChanged = Signal(str, float)

    def __init__(self, npImgCont, dim, xcoords, xycoords):
        super(FunctionWindow, self).__init__()
        self.npImgCont = npImgCont
        self.dim = dim
        self.xcoords = xcoords
        self.xycoords = xycoords

        self.setWindowTitle("Mapping/Shaping Function Window")
        self.setGeometry(20, 20, 600, 600)

        layout = QVBoxLayout()

        # Matplotlib figure 
        self.fig, self.ax = plt.subplots()
        self.ax.set_facecolor('black')
        self.fig.patch.set_facecolor('black')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        # Dropdown menu
        self.comboBox = QComboBox()
        self.comboBox.addItems(["y = x", "y = sin(x)", "y = pnoise1(x)", "y = almostIdentity(x)", "y = sin(x) 2D"])
        self.comboBox.currentIndexChanged.connect(self.do_function)
        layout.addWidget(self.comboBox)

        # Slider
        self.sliderLabel = QLabel("Parameter a.")
        layout.addWidget(self.sliderLabel)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1256)  # roughly 4PI * 100 for finer movement 
        self.slider.setValue(100)     # start value and will be divided by 100
        self.slider.valueChanged.connect(self.do_function)
        layout.addWidget(self.slider)

        # Finish setup.
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Start with something instead of empty plot.
        self.do_function() 


    def do_function(self):     
        self.ax.clear()
    
        a = self.slider.value() / 100.0    # make param a float in 2PI range
        # start fresh instead of modifying already drawn image
        self.npImgCont[:, :] = 1 

        if self.comboBox.currentText() == "y = x":
            y = self.xcoords * a 
            yplot = y

        elif self.comboBox.currentText() == "y = sin(x)":
            y = abs(np.sin(a*self.xcoords)) 
            yplot = y

        elif self.comboBox.currentText() == "y = pnoise1(x)":
            y = np.minimum(0.5+np.vectorize(noise.pnoise1)(self.xcoords+a, 4), 1)
            yplot = y

        elif self.comboBox.currentText() == "y = almostIdentity(x)":
            b = a/100
            if (b>0.007):
                b=0.007  # sanity check on parameter
            y = np.sqrt(self.xcoords * self.xcoords + b)
            yplot = y

        # 2D

        elif self.comboBox.currentText() == "y = sin(x) 2D":
            y = abs(np.sin(a*self.xcoords)) 
            yplot = y
            y = (abs(np.sin(a*self.xycoords[:, :, 0]))+abs(np.sin(a*self.xycoords[:, :, 1])))/2.0

        # Update npImgCont by broadcasting fcn output (y) over all of the rows of npImgCont (which should be 1s)
        np.multiply(self.npImgCont, y, out=self.npImgCont)
        self.ax.plot(self.xcoords, yplot, color='teal')
    
        # Set aspect ratio and axis limits
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, 1.1)
        self.ax.set_ylim(0, 1.1)
    
        # Add axes
        self.ax.axhline(0, color='gray', linewidth=0.5)
        self.ax.axvline(0, color='gray', linewidth=0.5)
    
        # Set axes labels
        self.ax.set_xlabel('X value', color='white')
        self.ax.set_ylabel('Y value', color='white')
    
        # Change tick color
        self.ax.tick_params(axis='both', colors='gray')
    
        self.canvas.draw()

        self.functionChanged.emit(self.comboBox.currentText(), a)



#########
# main
########


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ImageWindow()
    sys.exit(app.exec())

