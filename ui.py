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
    pt_start = mmf.Point2D(0, 0)
    pt_end = mmf.Point2D(0, 0)
    pt_prev_3d_start = mmf.Point3D(0, 0, 0)
    pt_unit_start = mmf.Point3D(0, 0, 0)

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

        self.fourth_button = QPushButton("Draw over", self)
        self.fourth_button.move(800, 140)
        self.fourth_button.clicked.connect(self.on_fourth_button_click)

        self.fifth_button = QPushButton("Project", self)
        self.fifth_button.move(800, 180)
        self.fifth_button.clicked.connect(self.on_fifth_button_click)

    def on_button_click(self):
        print("Button clicked!")
        angle = mmf.calc_rot_angle(mmf.Point2D(0, 0), mmf.Point2D(0, 1),  mmf.Point2D(1, 1))
        print(f"Angle: {angle}")

    def on_second_button_click(self):
        self.texture = miu.load_image_to_bitmap("resource/backgrounds/textureC2.bmp")
        self.texture  = self.texture.convert("RGBA")
        self.ref_quad.pts[0] = mmf.Point2D(318, 247)
        self.ref_quad.pts[1] = mmf.Point2D(326, 312)
        #self.ref_quad.pts[2] = mmf.Point2D(452, 303)
        #self.ref_quad.pts[3] = mmf.Point2D(418, 241)
        self.ref_quad.pts[3] = mmf.Point2D(452, 303)
        self.ref_quad.pts[2] = mmf.Point2D(418, 241)
        
        self.proj_matrix = mmf.compute_transfer_matrix(self.ref_quad.pts, 720, 576)
        print(self.proj_matrix)

    def on_third_button_click(self):
        #self.texture = miu.load_image_to_bitmap("resource/backgrounds/Untitled.bmp")
        #if self.texture is not None:
            # Convert PIL Image (BmpImageFile) to QImage
        #    image = self.texture.convert("RGBA")  # Ensure the image is in RGBA format
        #    data = image.tobytes("raw", "RGBA")
        #    qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
        #    self.image_label.setPixmap(QPixmap.fromImage(qimage).scaledToHeight(self.height()))  # Scale the image to fit the window height            
        #else:
        print("Failed to load texture image.")


    def on_fourth_button_click(self):
        # Draw the texture over the background image at the specified coordinates
        x = 100
        y = 100
        overlay_image_path = "resource/backgrounds/textureC2.bmp"
        self.draw_image(overlay_image_path, x, y)

    def on_fifth_button_click(self):
        # Project the texture onto the background image using the transformation matrix
        overlay_image_path = "resource/backgrounds/textureC2.bmp"
        overlay_image = miu.load_image_to_bitmap(overlay_image_path)
        if overlay_image is not None:
            overlay_image = overlay_image.convert("RGBA")
            pt2_start = mmf.Point2D(300, 300)
            pt2_end = mmf.Point2D(301, 301)
            #miu.project_texture_to_canvas(self.proj_matrix, pt2_start, pt2_end, pt3, pt_unit_start, 720, 576, overlay_image, self.image_label.pixmap().toImage())
            self.pt_prev_3d_start, self.pt_unit_start = miu.project_texture_to_canvas(self.proj_matrix, pt2_start, pt2_end, self.pt_prev_3d_start, self.pt_unit_start,  720, 576, overlay_image, self.image_label)
            #miu.project_texture_to_canvas(self.proj_matrix, pt2_start, pt2_end, pt3, pt_unit_start, 800, 600, overlay_image, self.image_label)

    def paintEvent(self, event):
        painter = QPainter(self)        
        painter.setBrush(QColor(64, 64, 64))  # Set brush to a dark grey color
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

    def draw_image(self, image_path, x, y):
        """
        Draw an image at the specified coordinates (x, y) over the image_label.

        :param image_path: Path to the image file to be drawn
        :param x: X coordinate where the image will be drawn
        :param y: Y coordinate where the image will be drawn
        """
        pixmap = self.image_label.pixmap()
        if pixmap is not None:
            image = pixmap.toImage()
            overlay_pixmap = QPixmap(image.width(), image.height())
            overlay_pixmap.fill(Qt.transparent)

            painter = QPainter(overlay_pixmap)
            painter.drawPixmap(0, 0, QPixmap.fromImage(image))  # Draw the existing image
            overlay_image = QPixmap(image_path)
            z = round(overlay_image.width() / 2)
            if not overlay_image.isNull():
                painter.drawPixmap(x - z, y, overlay_image)  # Draw the new image at the specified position
            painter.end()

            self.image_label.setPixmap(overlay_pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Canvas()
    window.show()
    sys.exit(app.exec_())