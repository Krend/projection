import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor 
import my_math_functions as mmf
import my_image_utility as miu
from PIL import Image

BACKGROUND_IMAGE = "resource/backgrounds/tankengine.bmp"

class Canvas(QMainWindow):
    ref_quad = mmf.Quad() # The 4 reference points which will be used to calculate the transformation matrix
    texture: Image # The texture image which will be projected onto the background
    proj_matrix =  mmf.Matrix4x4()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT User Interface")
        self.setGeometry(100, 100, 900, 600)  # x, y, width, height
        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 800, 600)  # Fill the window        
        self.load_background(BACKGROUND_IMAGE)  # Load the background image

        # Add a button at position (800, 20)
        self.button = QPushButton("Reset everything", self)
        self.button.move(800, 20)
        self.button.clicked.connect(self.on_button_click)

        # Add another button under it at position (800, 60)
        self.second_button = QPushButton("Calculate", self)
        self.second_button.move(800, 60)
        self.second_button.clicked.connect(self.on_second_button_click)

        self.third_button = QPushButton("Load texture", self)
        self.third_button.move(800, 100)
        self.third_button.clicked.connect(self.on_third_button_click)

    def on_button_click(self):
        print("Button clicked!")
        angle = mmf.calc_rot_angle(mmf.Point2D(0, 0), mmf.Point2D(0, 1),  mmf.Point2D(1, 1))
        print(f"Angle: {angle}")

    def on_second_button_click(self):
        self.ref_quad.pts[0] = mmf.Point2D(318, 247)
        self.ref_quad.pts[1] = mmf.Point2D(326, 312)
        #self.ref_quad.pts[2] = mmf.Point2D(452, 303)
        #self.ref_quad.pts[3] = mmf.Point2D(418, 241)
        self.ref_quad.pts[3] = mmf.Point2D(452, 303)
        self.ref_quad.pts[2] = mmf.Point2D(418, 241)
        
        self.proj_matrix = mmf.compute_transfer_matrix(self.ref_quad.pts, 720, 576)
        print(self.proj_matrix)

    def on_third_button_click(self):
        self.texture = miu.load_image_to_bitmap("resource/backgrounds/Untitled.bmp")
        if self.texture is not None:
            # Convert PIL Image (BmpImageFile) to QImage
            image = self.texture.convert("RGBA")  # Ensure the image is in RGBA format
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
            self.image_label.setPixmap(QPixmap.fromImage(qimage).scaledToHeight(self.height()))  # Scale the image to fit the window height
        else:
            print("Failed to load texture image.")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(Qt.blue)
        painter.drawRect(0, 0, self.width(), self.height())

    def mousePressEvent(self, event):        
        if event.button() == Qt.LeftButton:  # Check if the left mouse button was clicked
            x, y = event.x(), event.y()  # Get the x and y coordinates of the click            
            self.draw_ellipse(x, y, QColor(Qt.red))
            self.setWindowTitle(f"Mouse clicked at: ({x}, {y})")   

        if event.button() == Qt.RightButton:  # Check if the right mouse button was clicked  
            self.load_background(BACKGROUND_IMAGE)  # reset the image

    def mouseMoveEvent(self, event):        
        # Get the mouse position relative to the image_label
        label_pos = self.image_label.mapFromParent(event.pos())
        x, y = label_pos.x(), label_pos.y()

        # Check if the position is within the bounds of the image_label
        if 0 <= x < self.image_label.width() and 0 <= y < self.image_label.height():
            self.setWindowTitle(f"Mouse move over image_label at: ({x}, {y})")
            self.set_pixel(x, y, QColor(Qt.red))  # Set the pixel color to red at the position
        else:
            self.setWindowTitle("Mouse outside image_label")
                            
    def set_pixel(self, x: int, y: int, color: QColor):                
        pixmap = self.image_label.pixmap()
        if (pixmap is not None):
            image = pixmap.toImage()           

            if 0 <= x < image.width() and 0 <= y < image.height():
                image.setPixel(x, y, color.rgb())
                self.image_label.setPixmap(QPixmap.fromImage(image))

    def draw_cross(self, x: int, y: int, color: QColor, size: int = 5):
        """
        Draw a cross at the specified coordinates (x, y) with the given color and size.
        
        :param x: X coordinate of the center of the cross
        :param y: Y coordinate of the center of the cross
        :param color: Color of the cross
        :param size: Size of the cross
        """
        painter = QPainter(self.image_label.pixmap())
        painter.setPen(color)
        painter.drawLine(x - size, y, x + size, y)
        painter.drawLine(x, y - size, x, y + size)
        self.image_label.update()

    def draw_ellipse(self, x: int, y: int, color: QColor, size: int = 5):
        """
        Draw an ellipse at the specified coordinates (x, y) with the given color and size.
        
        :param x: X coordinate of the center of the ellipse
        :param y: Y coordinate of the center of the ellipse
        :param color: Color of the ellipse
        :param size: Size of the ellipse
        """
        painter = QPainter(self.image_label.pixmap())
        #painter.setPen(color)
        painter.setBrush(color)
        painter.drawEllipse(x - size, y - size, 2 * size, 2 * size)
        self.image_label.update()

    def load_background(self, file_path):
        """
        Load a background image from the specified file path.
        
        :param file_path: Path to the image file
        """
        pixmap = QPixmap(file_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaledToHeight(self.height()))  # Scale the image to fit the window height, otherwise it will be too big
        else:
            print(f"Failed to load image from {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Canvas()
    window.show()
    sys.exit(app.exec_())