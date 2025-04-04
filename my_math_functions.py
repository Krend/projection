from typing import List
from dataclasses import dataclass
# This file contains various mathematical functions used in the project.
# It includes functions for matrix operations, 3D to 2D projection, and 2D to 3D projection.
# The functions are designed to work with 3D points and matrices, and they handle various transformations.

#type definitions
@dataclass
class Point2D:
    x: float
    y: float

@dataclass
class Point3D:
    X: float
    Y: float
    Z: float

@dataclass
class Matrix3x3:
    m: List[float]
    def __init__(self):
        self.m = [0.0] * 9

@dataclass
class Matrix4x4:
    m: List[float]
    def __init__(self):
        self.m = [0.0] * 16

@dataclass
class Quad:
    pts:List[Point2D]
    def __init__(self):
        self.pts = [Point2D(0, 0) for _ in range(4)]        

#Utility functions for matrix operations

# matrix size of 9
def adjugate3x3(matrix: Matrix3x3):
    if len(matrix) != 9:
        return -1    
    return [matrix[4]*matrix[8]-matrix[5]*matrix[7], matrix[2]*matrix[7]-matrix[1]*matrix[8], matrix[1]*matrix[5]-matrix[2]*matrix[4],
            matrix[5]*matrix[6]-matrix[3]*matrix[8], matrix[0]*matrix[8]-matrix[2]*matrix[6], matrix[2]*matrix[3]-matrix[0]*matrix[5],
            matrix[3]*matrix[7]-matrix[4]*matrix[6], matrix[1]*matrix[6]-matrix[0]*matrix[7], matrix[0]*matrix[4]-matrix[1]*matrix[3] ]

# matrix size of 9
def determinant3x3(matrix: Matrix3x3):
    if len(matrix) != 9:
        return -1
    return matrix[0]*(matrix[4]*matrix[8]-matrix[5]*matrix[7]) - matrix[1]*(matrix[3]*matrix[8]-matrix[5]*matrix[6]) + matrix[2]*(matrix[3]*matrix[7]-matrix[4]*matrix[6])

# matrix size of 9
def inverse3x3(matrix: Matrix3x3):
    if len(matrix) != 9:
        return -1
    
    det = determinant3x3(matrix)

    if det == 0:
        return -1
    
    adj = adjugate3x3(matrix)

    for i in range(9):
        matrix[i] = matrix[i] / det

    return matrix

# matrix size of 16
def invers4x4(matrix: Matrix4x4):
    if len(matrix) != 16:
        return -1
    
    inv_matrix = Matrix4x4()
    inv_matrix[0] = ( matrix[5]  * matrix[10] * matrix[15] 
                     -matrix[5]  * matrix[11] * matrix[14] 
                     -matrix[9]  * matrix[6]  * matrix[15] 
                     +matrix[9]  * matrix[7]  * matrix[14] 
                     +matrix[13] * matrix[6]  * matrix[11] 
                     -matrix[13] * matrix[7]  * matrix[10])

    inv_matrix[4] = (-matrix[4]  * matrix[10] * matrix[15] 
                     +matrix[4]  * matrix[11] * matrix[14] 
                     +matrix[8]  * matrix[6]  * matrix[15] 
                     -matrix[8]  * matrix[7]  * matrix[14] 
                     -matrix[12] * matrix[6]  * matrix[11] 
                     +matrix[12] * matrix[7]  * matrix[10])

    inv_matrix[8] = ( matrix[4]  * matrix[9]  * matrix[15] 
                     -matrix[4]  * matrix[11] * matrix[13] 
                     -matrix[8]  * matrix[5]  * matrix[15] 
                     +matrix[8]  * matrix[7]  * matrix[13] 
                     +matrix[12] * matrix[5]  * matrix[11] 
                     -matrix[12] * matrix[7]  * matrix[9])

    inv_matrix[12] = (-matrix[4]  * matrix[9] * matrix[14] 
                      +matrix[4]  * matrix[10]* matrix[13] 
                      +matrix[8]  * matrix[5] * matrix[14] 
                      -matrix[8]  * matrix[6] * matrix[13] 
                      -matrix[12] * matrix[5] * matrix[10] 
                      +matrix[12] * matrix[6] * matrix[9])

    inv_matrix[1] = (-matrix[1]  * matrix[10] * matrix[15] 
                     +matrix[1]  * matrix[11] * matrix[14] 
                     +matrix[9]  * matrix[2] * matrix[15] 
                     -matrix[9]  * matrix[3] * matrix[14] 
                     -matrix[13] * matrix[2] * matrix[11] 
                     +matrix[13] * matrix[3] * matrix[10])

    inv_matrix[5] = ( matrix[0]  * matrix[10] * matrix[15] 
                     -matrix[0]  * matrix[11] * matrix[14] 
                     -matrix[8]  * matrix[2] * matrix[15] 
                     +matrix[8]  * matrix[3] * matrix[14] 
                     +matrix[12] * matrix[2] * matrix[11] 
                     -matrix[12] * matrix[3] * matrix[10])

    inv_matrix[9] = (-matrix[0]  * matrix[9] * matrix[15] 
                     +matrix[0]  * matrix[11] * matrix[13] 
                     +matrix[8]  * matrix[1] * matrix[15] 
                     -matrix[8]  * matrix[3] * matrix[13] 
                     -matrix[12] * matrix[1] * matrix[11] 
                     +matrix[12] * matrix[3] * matrix[9])

    inv_matrix[13] = ( matrix[0]  * matrix[9] * matrix[14] 
                      -matrix[0]  * matrix[10] * matrix[13] 
                      -matrix[8]  * matrix[1] * matrix[14] 
                      +matrix[8]  * matrix[2] * matrix[13] 
                      +matrix[12] * matrix[1] * matrix[10] 
                      -matrix[12] * matrix[2] * matrix[9])

    inv_matrix[2] = ( matrix[1]  * matrix[6] * matrix[15] 
                     -matrix[1]  * matrix[7] * matrix[14] 
                     -matrix[5]  * matrix[2] * matrix[15] 
                     +matrix[5]  * matrix[3] * matrix[14] 
                     +matrix[13] * matrix[2] * matrix[7] 
                     -matrix[13] * matrix[3] * matrix[6])

    inv_matrix[6] = (-matrix[0]  * matrix[6] * matrix[15] 
                     +matrix[0]  * matrix[7] * matrix[14] 
                     +matrix[4]  * matrix[2] * matrix[15] 
                     -matrix[4]  * matrix[3] * matrix[14] 
                     -matrix[12] * matrix[2] * matrix[7] 
                     +matrix[12] * matrix[3] * matrix[6])

    inv_matrix[10] = ( matrix[0]  * matrix[5] * matrix[15] 
                      -matrix[0]  * matrix[7] * matrix[13] 
                      -matrix[4]  * matrix[1] * matrix[15] 
                      +matrix[4]  * matrix[3] * matrix[13] 
                      +matrix[12] * matrix[1] * matrix[7] 
                      -matrix[12] * matrix[3] * matrix[5])

    inv_matrix[14] = (-matrix[0]  * matrix[5] * matrix[14] 
                      +matrix[0]  * matrix[6] * matrix[13] 
                      +matrix[4]  * matrix[1] * matrix[14] 
                      -matrix[4]  * matrix[2] * matrix[13] 
                      -matrix[12] * matrix[1] * matrix[6] 
                      +matrix[12] * matrix[2] * matrix[5])

    inv_matrix[3] = (-matrix[1] * matrix[6] * matrix[11] 
                     +matrix[1] * matrix[7] * matrix[10] 
                     +matrix[5] * matrix[2] * matrix[11] 
                     -matrix[5] * matrix[3] * matrix[10] 
                     -matrix[9] * matrix[2] * matrix[7] 
                     +matrix[9] * matrix[3] * matrix[6])

    inv_matrix[7] = ( matrix[0] * matrix[6] * matrix[11] 
                     -matrix[0] * matrix[7] * matrix[10] 
                     -matrix[4] * matrix[2] * matrix[11] 
                     +matrix[4] * matrix[3] * matrix[10] 
                     +matrix[8] * matrix[2] * matrix[7] 
                     -matrix[8] * matrix[3] * matrix[6])

    inv_matrix[11] = (-matrix[0] * matrix[5] * matrix[11] 
                      +matrix[0] * matrix[7] * matrix[9] 
                      +matrix[4] * matrix[1] * matrix[11] 
                      -matrix[4] * matrix[3] * matrix[9] 
                      -matrix[8] * matrix[1] * matrix[7] 
                      +matrix[8] * matrix[3] * matrix[5])

    inv_matrix[15] = ( matrix[0] * matrix[5] * matrix[10] 
                      -matrix[0] * matrix[6] * matrix[9] 
                      -matrix[4] * matrix[1] * matrix[10] 
                      +matrix[4] * matrix[2] * matrix[9] 
                      +matrix[8] * matrix[1] * matrix[6] 
                      -matrix[8] * matrix[2] * matrix[5])
    
    det = matrix[0] * inv_matrix[0] + matrix[1] * inv_matrix[4] + matrix[2] * inv_matrix[8] + matrix[3] * inv_matrix[12]

    if det == 0:
        return -1
    
    for i in range(16):
        inv_matrix[i] = inv_matrix[i] / det

    return inv_matrix

#matrix size of 9
def multiply3x3(matrix_a: Matrix3x3, matrix_b: Matrix3x3): 
    matrix_c = Matrix3x3()
    for i in range(3):
        for j in range(3):
            matrix_c[i * 3 + j] = sum(matrix_a[i * 3 + k] * matrix_b[k * 3 + j] for k in range(3))

    return matrix_c

# Advanced projections functions

def get_coeffs(x1: float, x2: float, x3: float, x4: float, y1: float, y2: float, y3: float, y4: float):
    denom = (x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2)
    
    if denom == 0:
        return -1, -1, -1   #todo: check if this can be a valid result
    
    g = ((-x1 * y3 + x3 * y1 - x4 * (y1 - y3) + y4 * (x1 - x3)) / denom)
    l = ((x2 * y3 - x3 * y2 + x4 * (y2 - y3) - y4 * (x2 - x3)) / denom)
    t = ((x1 * y2 - x2 * y1 + x4 * (y1 - y2) - y4 * (x1 - x2)) / denom)

    return g, l, t

# matrix size of 9
def source_to_dest(x_source: int, y_source: int, combined_matrix: Matrix3x3):
    xTmp = combined_matrix[0] * x_source + combined_matrix[1] * y_source + combined_matrix[2]
    yTmp = combined_matrix[3] * x_source + combined_matrix[4] * y_source + combined_matrix[5]
    zTmp = combined_matrix[6] * x_source + combined_matrix[7] * y_source + combined_matrix[8]

    if zTmp == 0:
        return -1, -1   #todo: check if this can be a valid result
       
    return (xTmp / zTmp), (yTmp / zTmp)

# Return the rectange which is around the quad
# point list size of 4
def quad_to_rect(quad_pts: Quad): 
    rect_pts = Quad()
    rect_pts[0].X = quad_pts[0].X
    rect_pts[1].X = quad_pts[0].X
    rect_pts[0].Y = quad_pts[0].Y
    rect_pts[1].Y = quad_pts[0].Y

    for i in range(1, 3):
        rect_pts[0].X = min(rect_pts[0].X, quad_pts[i].X)
        rect_pts[1].X = max(rect_pts[1].X, quad_pts[i].X)
        rect_pts[0].Y = min(rect_pts[0].Y, quad_pts[i].Y)
        rect_pts[1].Y = max(rect_pts[1].Y, quad_pts[i].Y)

    return rect_pts

# This function is used to project a 3D point onto a 2D plane using a transformation matrix.
# It takes a 3D point (Pt), a transformation matrix (matrix), and the width and height of the 2D plane as inputs.
# The function returns the projected 2D coordinates of the point.
# matrix size of 16
def project_3d_2d(pt3d: Point3D, width: int, height: int, matrix: Matrix4x4):
    x = matrix[0]  * pt3d.X + matrix[1]  * pt3d.Y + matrix[2]  * pt3d.Z + matrix[3]
    y = matrix[4]  * pt3d.X + matrix[5]  * pt3d.Y + matrix[6]  * pt3d.Z + matrix[7]
    w = matrix[12] * pt3d.X + matrix[13] * pt3d.Y + matrix[14] * pt3d.Z + matrix[15]
    
    if w == 0:
        return -1, -1   #todo: check if this can be a valid result
        
    return (width * (x / w + 1.0) * 0.5), (height * (y / w + 1.0) * 0.5)

# This function is used to project a 2D point onto a 3D space using an inverse transformation matrix.
# It takes a 2D point (Pt), an inverse transformation matrix (inv_matrix), and the width and height of the 2D plane as inputs.
# The function returns the projected 3D coordinates of the point.
# matrix size of 16
def project_2d_3d(pt2d: Point2D, width: int, height: int, inv_matrix: Matrix4x4):
    x = (pt2d.x / width) * 2.0 -1.0    
    y = (pt2d.y / height) * 2.0 -1.0
    z = 0   # No cordinate for z, since we are using a 2D point

    x2 = inv_matrix[0]  * x + inv_matrix[1]  * y + inv_matrix[2]  * z + inv_matrix[3]
    y2 = inv_matrix[4]  * x + inv_matrix[5]  * y + inv_matrix[6]  * z + inv_matrix[7]
    z2 = inv_matrix[8]  * x + inv_matrix[9]  * y + inv_matrix[10] * z + inv_matrix[11]
    w =  inv_matrix[12] * x + inv_matrix[13] * y + inv_matrix[14] * z + inv_matrix[15]

    if w == 0:
        return -1, -1, -1   #todo: check if this can be a valid result
    
    return (x2 / w), (y2 / w), (z2 / w)

