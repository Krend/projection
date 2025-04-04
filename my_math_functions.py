from typing import List
from dataclasses import dataclass
# This file contains various mathematical functions used in the project.
# It includes functions for matrix.m operations, 3D to 2D projection, and 2D to 3D projection.
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
    pts: List[Point2D]
    def __init__(self):
        self.pts = [Point2D(0, 0) for _ in range(4)]

@dataclass
class Rect:
    pt0: Point2D
    pt1: Point2D
    def __init__(self):
        self.pt0 = Point2D(0, 0)
        self.pt1 = Point2D(0, 0)

#Utility functions for matrix.m operations

# matrix.m size of 9
def adjugate3x3(matrix: Matrix3x3):
    if len(matrix.m) != 9:
        return -1    
    return [matrix.m[4]*matrix.m[8]-matrix.m[5]*matrix.m[7], matrix.m[2]*matrix.m[7]-matrix.m[1]*matrix.m[8], matrix.m[1]*matrix.m[5]-matrix.m[2]*matrix.m[4],
            matrix.m[5]*matrix.m[6]-matrix.m[3]*matrix.m[8], matrix.m[0]*matrix.m[8]-matrix.m[2]*matrix.m[6], matrix.m[2]*matrix.m[3]-matrix.m[0]*matrix.m[5],
            matrix.m[3]*matrix.m[7]-matrix.m[4]*matrix.m[6], matrix.m[1]*matrix.m[6]-matrix.m[0]*matrix.m[7], matrix.m[0]*matrix.m[4]-matrix.m[1]*matrix.m[3] ]

# matrix.m size of 9
def determinant3x3(matrix: Matrix3x3):
    if len(matrix.m) != 9:
        return -1
    return matrix.m[0]*(matrix.m[4]*matrix.m[8]-matrix.m[5]*matrix.m[7]) - matrix.m[1]*(matrix.m[3]*matrix.m[8]-matrix.m[5]*matrix.m[6]) + matrix.m[2]*(matrix.m[3]*matrix.m[7]-matrix.m[4]*matrix.m[6])

# matrix.m size of 9
def inverse3x3(matrix: Matrix3x3):
    if len(matrix.m) != 9:
        return -1
    
    det = determinant3x3(matrix.m)

    if det == 0:
        return -1
    
    adj = adjugate3x3(matrix.m)

    for i in range(9):
        matrix.m[i] = matrix.m[i] / det

    return matrix

# matrix.m size of 16
def invers4x4(matrix: Matrix4x4):
    if len(matrix.m) != 16:
        return -1
    
    inv_matrix = Matrix4x4()
    inv_matrix.m[0] = ( matrix.m[5]  * matrix.m[10] * matrix.m[15] 
                     -matrix.m[5]  * matrix.m[11] * matrix.m[14] 
                     -matrix.m[9]  * matrix.m[6]  * matrix.m[15] 
                     +matrix.m[9]  * matrix.m[7]  * matrix.m[14] 
                     +matrix.m[13] * matrix.m[6]  * matrix.m[11] 
                     -matrix.m[13] * matrix.m[7]  * matrix.m[10])

    inv_matrix.m[4] = (-matrix.m[4]  * matrix.m[10] * matrix.m[15] 
                     +matrix.m[4]  * matrix.m[11] * matrix.m[14] 
                     +matrix.m[8]  * matrix.m[6]  * matrix.m[15] 
                     -matrix.m[8]  * matrix.m[7]  * matrix.m[14] 
                     -matrix.m[12] * matrix.m[6]  * matrix.m[11] 
                     +matrix.m[12] * matrix.m[7]  * matrix.m[10])

    inv_matrix.m[8] = ( matrix.m[4]  * matrix.m[9]  * matrix.m[15] 
                     -matrix.m[4]  * matrix.m[11] * matrix.m[13] 
                     -matrix.m[8]  * matrix.m[5]  * matrix.m[15] 
                     +matrix.m[8]  * matrix.m[7]  * matrix.m[13] 
                     +matrix.m[12] * matrix.m[5]  * matrix.m[11] 
                     -matrix.m[12] * matrix.m[7]  * matrix.m[9])

    inv_matrix.m[12] = (-matrix.m[4]  * matrix.m[9] * matrix.m[14] 
                      +matrix.m[4]  * matrix.m[10]* matrix.m[13] 
                      +matrix.m[8]  * matrix.m[5] * matrix.m[14] 
                      -matrix.m[8]  * matrix.m[6] * matrix.m[13] 
                      -matrix.m[12] * matrix.m[5] * matrix.m[10] 
                      +matrix.m[12] * matrix.m[6] * matrix.m[9])

    inv_matrix.m[1] = (-matrix.m[1]  * matrix.m[10] * matrix.m[15] 
                     +matrix.m[1]  * matrix.m[11] * matrix.m[14] 
                     +matrix.m[9]  * matrix.m[2] * matrix.m[15] 
                     -matrix.m[9]  * matrix.m[3] * matrix.m[14] 
                     -matrix.m[13] * matrix.m[2] * matrix.m[11] 
                     +matrix.m[13] * matrix.m[3] * matrix.m[10])

    inv_matrix.m[5] = ( matrix.m[0]  * matrix.m[10] * matrix.m[15] 
                     -matrix.m[0]  * matrix.m[11] * matrix.m[14] 
                     -matrix.m[8]  * matrix.m[2] * matrix.m[15] 
                     +matrix.m[8]  * matrix.m[3] * matrix.m[14] 
                     +matrix.m[12] * matrix.m[2] * matrix.m[11] 
                     -matrix.m[12] * matrix.m[3] * matrix.m[10])

    inv_matrix.m[9] = (-matrix.m[0]  * matrix.m[9] * matrix.m[15] 
                     +matrix.m[0]  * matrix.m[11] * matrix.m[13] 
                     +matrix.m[8]  * matrix.m[1] * matrix.m[15] 
                     -matrix.m[8]  * matrix.m[3] * matrix.m[13] 
                     -matrix.m[12] * matrix.m[1] * matrix.m[11] 
                     +matrix.m[12] * matrix.m[3] * matrix.m[9])

    inv_matrix.m[13] = ( matrix.m[0]  * matrix.m[9] * matrix.m[14] 
                      -matrix.m[0]  * matrix.m[10] * matrix.m[13] 
                      -matrix.m[8]  * matrix.m[1] * matrix.m[14] 
                      +matrix.m[8]  * matrix.m[2] * matrix.m[13] 
                      +matrix.m[12] * matrix.m[1] * matrix.m[10] 
                      -matrix.m[12] * matrix.m[2] * matrix.m[9])

    inv_matrix.m[2] = ( matrix.m[1]  * matrix.m[6] * matrix.m[15] 
                     -matrix.m[1]  * matrix.m[7] * matrix.m[14] 
                     -matrix.m[5]  * matrix.m[2] * matrix.m[15] 
                     +matrix.m[5]  * matrix.m[3] * matrix.m[14] 
                     +matrix.m[13] * matrix.m[2] * matrix.m[7] 
                     -matrix.m[13] * matrix.m[3] * matrix.m[6])

    inv_matrix.m[6] = (-matrix.m[0]  * matrix.m[6] * matrix.m[15] 
                     +matrix.m[0]  * matrix.m[7] * matrix.m[14] 
                     +matrix.m[4]  * matrix.m[2] * matrix.m[15] 
                     -matrix.m[4]  * matrix.m[3] * matrix.m[14] 
                     -matrix.m[12] * matrix.m[2] * matrix.m[7] 
                     +matrix.m[12] * matrix.m[3] * matrix.m[6])

    inv_matrix.m[10] = ( matrix.m[0]  * matrix.m[5] * matrix.m[15] 
                      -matrix.m[0]  * matrix.m[7] * matrix.m[13] 
                      -matrix.m[4]  * matrix.m[1] * matrix.m[15] 
                      +matrix.m[4]  * matrix.m[3] * matrix.m[13] 
                      +matrix.m[12] * matrix.m[1] * matrix.m[7] 
                      -matrix.m[12] * matrix.m[3] * matrix.m[5])

    inv_matrix.m[14] = (-matrix.m[0]  * matrix.m[5] * matrix.m[14] 
                      +matrix.m[0]  * matrix.m[6] * matrix.m[13] 
                      +matrix.m[4]  * matrix.m[1] * matrix.m[14] 
                      -matrix.m[4]  * matrix.m[2] * matrix.m[13] 
                      -matrix.m[12] * matrix.m[1] * matrix.m[6] 
                      +matrix.m[12] * matrix.m[2] * matrix.m[5])

    inv_matrix.m[3] = (-matrix.m[1] * matrix.m[6] * matrix.m[11] 
                     +matrix.m[1] * matrix.m[7] * matrix.m[10] 
                     +matrix.m[5] * matrix.m[2] * matrix.m[11] 
                     -matrix.m[5] * matrix.m[3] * matrix.m[10] 
                     -matrix.m[9] * matrix.m[2] * matrix.m[7] 
                     +matrix.m[9] * matrix.m[3] * matrix.m[6])

    inv_matrix.m[7] = ( matrix.m[0] * matrix.m[6] * matrix.m[11] 
                     -matrix.m[0] * matrix.m[7] * matrix.m[10] 
                     -matrix.m[4] * matrix.m[2] * matrix.m[11] 
                     +matrix.m[4] * matrix.m[3] * matrix.m[10] 
                     +matrix.m[8] * matrix.m[2] * matrix.m[7] 
                     -matrix.m[8] * matrix.m[3] * matrix.m[6])

    inv_matrix.m[11] = (-matrix.m[0] * matrix.m[5] * matrix.m[11] 
                      +matrix.m[0] * matrix.m[7] * matrix.m[9] 
                      +matrix.m[4] * matrix.m[1] * matrix.m[11] 
                      -matrix.m[4] * matrix.m[3] * matrix.m[9] 
                      -matrix.m[8] * matrix.m[1] * matrix.m[7] 
                      +matrix.m[8] * matrix.m[3] * matrix.m[5])

    inv_matrix.m[15] = ( matrix.m[0] * matrix.m[5] * matrix.m[10] 
                      -matrix.m[0] * matrix.m[6] * matrix.m[9] 
                      -matrix.m[4] * matrix.m[1] * matrix.m[10] 
                      +matrix.m[4] * matrix.m[2] * matrix.m[9] 
                      +matrix.m[8] * matrix.m[1] * matrix.m[6] 
                      -matrix.m[8] * matrix.m[2] * matrix.m[5])
    
    det = matrix.m[0] * inv_matrix.m[0] + matrix.m[1] * inv_matrix.m[4] + matrix.m[2] * inv_matrix.m[8] + matrix.m[3] * inv_matrix.m[12]

    if det == 0:
        return -1
    
    for i in range(16):
        inv_matrix.m[i] = inv_matrix.m[i] / det

    return inv_matrix

#matrix.m size of 9
def multiply3x3(matrix_a: Matrix3x3, matrix_b: Matrix3x3): 
    matrix_c = Matrix3x3()
    for i in range(3):
        for j in range(3):
            matrix_c.m[i * 3 + j] = sum(matrix_a.m[i * 3 + k] * matrix_b.m[k * 3 + j] for k in range(3))

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

# matrix.m size of 9
def source_to_dest(x_source: int, y_source: int, combined_matrix: Matrix3x3):
    xTmp = combined_matrix.m[0] * x_source + combined_matrix.m[1] * y_source + combined_matrix.m[2]
    yTmp = combined_matrix.m[3] * x_source + combined_matrix.m[4] * y_source + combined_matrix.m[5]
    zTmp = combined_matrix.m[6] * x_source + combined_matrix.m[7] * y_source + combined_matrix.m[8]

    if zTmp == 0:
        return -1, -1   #todo: check if this can be a valid result
       
    return (xTmp / zTmp), (yTmp / zTmp)

# Return the rectange which is around the quad
# quad.pts size of 4
def quad_to_rect(quad: Quad): 
    if len(quad) != 4:
        return -1
    rect = Rect()
    rect.pt0.X = quad.pts[0].X
    rect.pt1.X = quad.pts[0].X
    rect.pt0.Y = quad.pts[0].Y
    rect.pt1.Y = quad.pts[0].Y

    for i in range(1, 3):
        rect.pt0.X = min(rect.pt0.X, quad.pts[i].X)
        rect.pt1.X = max(rect.pt1.X, quad.pts[i].X)
        rect.pt0.Y = min(rect.pt0.Y, quad.pts[i].Y)
        rect.pt1.Y = max(rect.pt1.Y, quad.pts[i].Y)

    return rect

# This function is used to project a 3D point onto a 2D plane using a transformation matrix.m.
# It takes a 3D point (Pt), a transformation matrix.m (matrix.m), and the width and height of the 2D plane as inputs.
# The function returns the projected 2D coordinates of the point.
# matrix.m size of 16
def project_3d_2d(pt3d: Point3D, width: int, height: int, matrix: Matrix4x4):
    x = matrix.m[0]  * pt3d.X + matrix.m[1]  * pt3d.Y + matrix.m[2]  * pt3d.Z + matrix.m[3]
    y = matrix.m[4]  * pt3d.X + matrix.m[5]  * pt3d.Y + matrix.m[6]  * pt3d.Z + matrix.m[7]
    w = matrix.m[12] * pt3d.X + matrix.m[13] * pt3d.Y + matrix.m[14] * pt3d.Z + matrix.m[15]
    
    if w == 0:
        return -1, -1   #todo: check if this can be a valid result
        
    return (width * (x / w + 1.0) * 0.5), (height * (y / w + 1.0) * 0.5)

# This function is used to project a 2D point onto a 3D space using an inverse transformation matrix.m.
# It takes a 2D point (Pt), an inverse transformation matrix.m (inv_matrix.m), and the width and height of the 2D plane as inputs.
# The function returns the projected 3D coordinates of the point.
# matrix.m size of 16
def project_2d_3d(pt2d: Point2D, width: int, height: int, inv_matrix: Matrix4x4):
    x = (pt2d.x / width) * 2.0 -1.0    
    y = (pt2d.y / height) * 2.0 -1.0
    z = 0   # No cordinate for z, since we are using a 2D point

    x2 = inv_matrix.m[0]  * x + inv_matrix.m[1]  * y + inv_matrix.m[2]  * z + inv_matrix.m[3]
    y2 = inv_matrix.m[4]  * x + inv_matrix.m[5]  * y + inv_matrix.m[6]  * z + inv_matrix.m[7]
    z2 = inv_matrix.m[8]  * x + inv_matrix.m[9]  * y + inv_matrix.m[10] * z + inv_matrix.m[11]
    w =  inv_matrix.m[12] * x + inv_matrix.m[13] * y + inv_matrix.m[14] * z + inv_matrix.m[15]

    if w == 0:
        return -1, -1, -1   #todo: check if this can be a valid result
    
    return (x2 / w), (y2 / w), (z2 / w)

