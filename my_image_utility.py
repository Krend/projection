from PIL import Image
from typing import List
import my_math_functions as mmf
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt

def load_image_to_bitmap(file_path):
    """
    Load an image file into a bitmap.

    :param file_path: Path to the image file
    :return: Image object (bitmap)
    """
    try:
        image = Image.open(file_path)        
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
def project_texture(tex_source: Image, tex_target: Image, bounds_rect: mmf.Rect, combined_matrix: mmf.Matrix3x3):
    """
    Project a texture (im_source) onto a target image (im_target) using a projection matrix.

    :param im_source: Source image (texture)
    :param im_target: Target image
    :param bounds_rect: Rectangle defining the bounds for projection
    :param combined_matrix: Combined 3x3 matrix
    """

    x_min  = round(max(0, min(bounds_rect.pt0.x, tex_target.width - 1)))
    y_min  = round(max(0, min(bounds_rect.pt0.y, tex_target.height - 1)))
    x_max  = round(max(0, min(bounds_rect.pt1.x, tex_target.width - 1)))
    y_max  = round(max(0, min(bounds_rect.pt1.y, tex_target.height - 1)))

    for j in range(y_min, y_max - 1):
        for i in range(x_min, x_max - 1):
            # Get the pixel coordinates in the source image
            x_source, y_source = mmf.source_to_dest(i, j, combined_matrix)
            if 0 <= x_source < tex_source.width -1 and 0 <= y_source < tex_source.height -1:
                color = tex_source.getpixel((round(x_source), round(y_source)))  # Pass coordinates as a tuple
                tex_target.putpixel((i, j), color)            
    return tex_target

def project_texture_to_canvas(proj_matrix: mmf.Matrix4x4, pt_start: mmf.Point2D, pt_end: mmf.Point2D, pt_prev_3d_start: mmf.Point3D, pt_unit_start: mmf.Point3D, width: int, height: int, tex_in: Image, canvas: QLabel):
    inv_matrix = mmf.invers4x4(proj_matrix)

    pt2_in_list: List[mmf.Point2D] = []
    pt2_in_list.append(pt_start)
    pt2_in_list.append(pt_end)
    pt3_out_list = mmf.list_reverse_project_2d_3d(pt2_in_list, width, height, inv_matrix)

    if pt3_out_list[0] == pt_prev_3d_start:
        pt3_out_list[0] = pt_unit_start

    rec_3d, unit_rec = mmf.line_to_unit_square_3d_3d(pt3_out_list[0], pt3_out_list[1])
    pt_prev_3d_start = pt3_out_list[1]
    pt2_quad = mmf.list_project_3d_2d(rec_3d, width, height, proj_matrix)

    l, g, t = mmf.get_coeffs(pt2_quad[0].x, pt2_quad[1].x, pt2_quad[2].x, pt2_quad[3].x,
                             pt2_quad[0].y, pt2_quad[1].y, pt2_quad[2].y, pt2_quad[3].y)
    matrix_b = mmf.Matrix3x3()
    matrix_b.m = [l * pt2_quad[0].x, g * pt2_quad[1].x, t * pt2_quad[2].x,
                  l * pt2_quad[0].y, g * pt2_quad[1].y, t * pt2_quad[2].y,
                  l,                 g,                 t]

    l, g, t = mmf.get_coeffs(0, 0, tex_in.width, tex_in.width,
                             0, tex_in.height, tex_in.height, 0)
    matrix_a = mmf.Matrix3x3()
    matrix_a.m = [0, 0,                 t * tex_in.width,
                  0, g * tex_in.height, t * tex_in.height,
                  l, g,                 t]
    matrix_b = mmf.inverse3x3(matrix_b)

    matrix_c = mmf.multiply3x3(matrix_a, matrix_b)

    quad = mmf.Quad()
    quad.pts = pt2_quad

    rec = mmf.quad_to_rect(quad)

    tex_out = Image.new(tex_in.mode, (canvas.width(), canvas.height()), (0, 0, 0, 0))
    project_texture(tex_in, tex_out, rec, matrix_c)

    # Draw the texture onto the canvas
    draw_onto_qlabel(canvas, tex_out)
    #draw_onto_qlabel(canvas, tex_in)
    draw_bounding_rectangle(canvas, rec)
    draw_bounding_quad(canvas, quad)    


def draw_onto_qlabel(canvas: QLabel, image_in: Image):

    pixmap = canvas.pixmap()
    if pixmap is not None:
        image = pixmap.toImage()
        overlay_pixmap = QPixmap(image.width(), image.height())
        overlay_pixmap.fill(Qt.transparent)

        painter = QPainter(overlay_pixmap)
        painter.drawPixmap(0, 0, QPixmap.fromImage(image))  # Draw the existing image
        #overlay_image = QPixmap(image_in)
        overlay_image = QPixmap.fromImage(QImage(image_in.tobytes(), image_in.width, image_in.height, QImage.Format_RGBA8888))

        if not overlay_image.isNull():
            painter.drawPixmap(0, 0, overlay_image)  # Draw the new image at the specified position
        painter.end()
        canvas.setPixmap(overlay_pixmap)

def draw_bounding_rectangle(canvas: QLabel, rect: mmf.Rect):
    """
    Draw a bounding rectangle on the canvas.

    :param canvas: QLabel to draw on
    :param rect: Rectangle to draw
    """
    painter = QPainter(canvas.pixmap())
    painter.setPen(Qt.red)
    painter.drawRect(round(rect.pt0.x), round(rect.pt0.y), round(rect.pt1.x - rect.pt0.x), round(rect.pt1.y - rect.pt0.y))
    
    painter.end()

def draw_bounding_quad(canvas: QLabel, quad: mmf.Quad):
    """
    Draw a bounding quad on the canvas.

    :param canvas: QLabel to draw on
    :param quad: Quad to draw
    """
    painter = QPainter(canvas.pixmap())
    painter.setPen(Qt.red)
    
    # Draw lines between each point in the quad
    for i in range(len(quad.pts)):
        start_point = quad.pts[i]
        end_point = quad.pts[(i + 1) % len(quad.pts)]  # Wrap around to the first point
        painter.drawLine(round(start_point.x), round(start_point.y), round(end_point.x), round(end_point.y))
    
    painter.end()