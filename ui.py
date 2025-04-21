import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor 
import my_math_functions as mmf
import my_image_utility as miu
from PIL import Image

BACKGROUND_IMAGE = "resource/backgrounds/railways.bmp"

class Canvas(QMainWindow):
    ref_quad = mmf.Quad() # The 4 reference points which will be used to calculate the transformation matrix
    texture: Image # The texture image which will be projected onto the background
    proj_matrix =  mmf.Matrix4x4()
    pt_start = mmf.Point2D(0, 0)
    pt_end = mmf.Point2D(0, 0)
    pt_prev_3d_start = mmf.Point3D(0, 0, 0)
    pt_unit_start = mmf.Point3D(0, 0, 0)
    calc_done = False

    def __init__(self):
        super().__init__()
        self.setWindowTitle("QT User Interface")
        self.setGeometry(100, 100, 9000, 600)  # x, y, width, height
        self.setFixedSize(QSize(900, 600)); # resizing messes up the relative coordinates
        self.image_label = QLabel(self)
        self.image_label.setGeometry(0, 0, 800, 600)  # Fill the window        
        self.load_background(BACKGROUND_IMAGE)  # Load the background image
        
        self.btn_calc = QPushButton("Calculate", self)
        self.btn_calc.move(800, 60)
        self.btn_calc.clicked.connect(self.on_btn_calc_click)
        
    def on_btn_calc_click(self):
        self.texture = miu.load_image_from_file("resource/backgrounds/arrow_texture.bmp")
        self.texture  = self.texture.convert("RGBA")
        self.ref_quad.pts[0] = mmf.Point2D(318, 247)
        self.ref_quad.pts[1] = mmf.Point2D(326, 312)        
        self.ref_quad.pts[3] = mmf.Point2D(452, 303)
        self.ref_quad.pts[2] = mmf.Point2D(418, 241)
        
        self.proj_matrix = mmf.compute_transfer_matrix(self.ref_quad.pts, 720, 576)
        print(self.proj_matrix)
        self.calc_done = True

    def paintEvent(self, event):
        painter = QPainter(self)        
        painter.setBrush(QColor(64, 64, 64))  # Set brush to a dark grey color
        painter.drawRect(0, 0, self.width(), self.height())

    #def mousePressEvent(self, event):        
        #if event.button() == Qt.LeftButton:  # Check if the left mouse button was clicked
            #x, y = event.x(), event.y()  # Get the x and y coordinates of the click            
            #self.draw_ellipse(x, y, QColor(Qt.red))
            #self.setWindowTitle(f"Mouse clicked at: ({x}, {y})")   

        #if event.button() == Qt.RightButton:  # Check if the right mouse button was clicked  
            #self.load_background(BACKGROUND_IMAGE)  # reset the image

    def mouseMoveEvent(self, event):
        """
        Handles the mouse move event for the UI.
        This method is triggered when the mouse is moved within the widget. It calculates
        the mouse position relative to the `image_label` and performs actions based on
        whether the mouse is within the bounds of the `image_label`.
        Args:
            event (QMouseEvent): The mouse event containing information about the mouse position.
        Behavior:
            - Maps the mouse position from the parent widget to the `image_label`.
            - Checks if the mouse position is within the bounds of the `image_label`.
            - If within bounds:
                - Updates the `pt_end` attribute with the current mouse position.
                - Resets the background image using `load_background`.
                - Projects the texture to the canvas using `miu.project_texture_to_canvas`.
                - Updates the `pt_start` attribute with the current mouse position.
            - If outside bounds:
                - Updates the window title to indicate the mouse is outside the `image_label`.
        """
        if not self.calc_done:
            return
        # Get the mouse position relative to the image_label
        label_pos = self.image_label.mapFromParent(event.pos())
        x, y = label_pos.x(), label_pos.y()
        
        #todo: might wrap this part into a try/except block to not crash the program when a calculation is invalid

        # Check if the position is within the bounds of the image_label
        if 0 <= x < self.image_label.width() and 0 <= y < self.image_label.height():
            self.pt_end = mmf.Point2D(x, y)
            self.load_background(BACKGROUND_IMAGE)  # reset the image            
            self.pt_prev_3d_start, self.pt_unit_start = miu.project_texture_to_canvas(self.proj_matrix, self.pt_start, self.pt_end, self.pt_prev_3d_start, self.pt_unit_start, 720, 576, self.texture, self.image_label)
            self.pt_start = self.pt_end

        else:
            self.setWindowTitle("Mouse outside image_label")
                            
    def set_pixel(self, x: int, y: int, color: QColor):                
        pixmap = self.image_label.pixmap()
        if (pixmap is not None):
            image = pixmap.toImage()           

            if 0 <= x < image.width() and 0 <= y < image.height():
                image.setPixel(x, y, color.rgb())
                self.image_label.setPixmap(QPixmap.fromImage(image))
                
    def draw_ellipse(self, x: int, y: int, color: QColor, size: int = 5):
        """
        Draw an ellipse at the specified coordinates (x, y) with the given color and size.
        
        :param x: X coordinate of the center of the ellipse
        :param y: Y coordinate of the center of the ellipse
        :param color: Color of the ellipse
        :param size: Size of the ellipse
        """
        painter = QPainter(self.image_label.pixmap())        
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