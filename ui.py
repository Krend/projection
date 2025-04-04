import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt

class Canvas(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT User Interface")
        self.setGeometry(100, 100, 800, 600)  # x, y, width, height
        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 800, 600)  # Fill the window
        self.image_label.setPixmap(QPixmap("Untitled.bmp").scaled(800, 600, Qt.KeepAspectRatio)) #todo: add image path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(Qt.blue)
        painter.drawRect(0, 0, self.width(), self.height())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:  # Check if the left mouse button was clicked
            x, y = event.x(), event.y()  # Get the x and y coordinates of the click
            #print(f"Mouse clicked at: ({x}, {y})")
            self.setWindowTitle(f"Mouse clicked at: ({x}, {y})")

    def mouseMoveEvent(self,event):
        #if event.button() == Qt.LeftButton:  # Check if the left mouse button was clicked
            x, y = event.x(), event.y()  # Get the x and y coordinates of the click
            #print(f"Mouse clicked at: ({x}, {y})")
            self.setWindowTitle(f"Mouse move at: ({x}, {y})")
            

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Canvas()
    window.show()
    sys.exit(app.exec_())