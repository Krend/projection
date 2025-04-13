from typing import List
from dataclasses import dataclass
import random
import math
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
    x: float
    y: float
    z: float

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

#Utility functions for matrix operations

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
    if len(quad.pts) != 4:
        return -1
    rect = Rect()
    rect.pt0.x = quad.pts[0].x
    rect.pt1.x = quad.pts[0].x
    rect.pt0.y = quad.pts[0].y
    rect.pt1.y = quad.pts[0].y

    for i in range(1, 3):
        rect.pt0.x = min(rect.pt0.x, quad.pts[i].x)
        rect.pt1.x = max(rect.pt1.x, quad.pts[i].x)
        rect.pt0.y = min(rect.pt0.y, quad.pts[i].y)
        rect.pt1.y = max(rect.pt1.y, quad.pts[i].y)

    return rect

# This function is used to project a 3D point onto a 2D plane using a transformation matrix.m.
# It takes a 3D point (Pt), a transformation matrix.m (matrix.m), and the width and height of the 2D plane as inputs.
# The function returns the projected 2D coordinates of the point.
# matrix.m size of 16
def project_3d_2d(pt3d: Point3D, width: int, height: int, matrix: Matrix4x4):
    x = matrix.m[0]  * pt3d.x + matrix.m[1]  * pt3d.y + matrix.m[2]  * pt3d.z + matrix.m[3]
    y = matrix.m[4]  * pt3d.x + matrix.m[5]  * pt3d.y + matrix.m[6]  * pt3d.z + matrix.m[7]
    # z axis is not used in 2D projection
    w = matrix.m[12] * pt3d.x + matrix.m[13] * pt3d.y + matrix.m[14] * pt3d.z + matrix.m[15]
    
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
    z = 0   # No value for z, since we are using a 2D point (I'll leave the rest of the code as is for clarity)

    x2 = inv_matrix.m[0]  * x + inv_matrix.m[1]  * y + inv_matrix.m[2]  * z + inv_matrix.m[3]
    y2 = inv_matrix.m[4]  * x + inv_matrix.m[5]  * y + inv_matrix.m[6]  * z + inv_matrix.m[7]
    z2 = inv_matrix.m[8]  * x + inv_matrix.m[9]  * y + inv_matrix.m[10] * z + inv_matrix.m[11]
    w =  inv_matrix.m[12] * x + inv_matrix.m[13] * y + inv_matrix.m[14] * z + inv_matrix.m[15]

    if w == 0:
        return -1, -1, -1   #todo: check if this can be a valid result
    
    return (x2 / w), (y2 / w), (z2 / w)

def point_dist_2d(pt1: Point2D, pt2: Point2D):
    #return ((pt1.x - pt2.x) ** 2 + (pt1.y - pt2.y) ** 2) ** 0.5
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    return dx * dx + dy * dy

def eval_dist(matrix: Matrix4x4, pt3d_list : List[Point3D], pt2d_list: List[Point2D], width: int, height: int):
    if len(pt3d_list) != len(pt2d_list):
        return -1
    
    dist = 0
    for i in range(len(pt3d_list)):
        pt2d = Point2D(0, 0)
        pt2d.x, pt2d.y = project_3d_2d(pt3d_list[i], width, height, matrix)
        dist += point_dist_2d(pt2d, pt2d_list[i])

    return dist

def perturb_matrix(matrix: Matrix4x4, perturbation: float):
    perturbed_matrix = Matrix4x4()
    for i in range(len(matrix.m)):
        perturbed_matrix.m[i] = matrix.m[i]

    i = random.randrange(0, 16)    
    perturbed_matrix.m[i] += random.uniform(-perturbation, perturbation)
    return perturbed_matrix

def approximate_matrix(matrix: Matrix4x4, pt3d_list : List[Point3D], pt2d_list: List[Point2D], width: int, height: int, n:  int, perturbation: float):
    if len(pt3d_list) != len(pt2d_list):
        return -1
    
    best_matrix = Matrix4x4()
    for i in range(len(matrix.m)):
        best_matrix.m[i] = matrix.m[i]
    best_dist = eval_dist(best_matrix, pt3d_list, pt2d_list, width, height)
        
    for i in range(n):  # Number of iterations
        perturbed_matrix = perturb_matrix(best_matrix, perturbation) 
        dist = eval_dist(perturbed_matrix, pt3d_list, pt2d_list, width, height)
        
        if dist < best_dist:
            best_dist = dist
            for j in range(len(best_matrix.m)):
                best_matrix.m[j] = perturbed_matrix.m[j]

    return best_matrix

def compute_transfer_matrix(pt2d_list: List[Point2D], width: int, height: int):
    matrix = Matrix4x4()
    # Initialize the matrix with identity values 
    # We will distort this until we get the best fit for the points
    matrix.m[0] = 1.0
    matrix.m[5] = 1.0
    matrix.m[10] = 1.0
    matrix.m[15] = 1.0

    #todo: check if it's required to swap [2] and [3] in pt2d_list (will depend on my input data in this implementation)

    pt3d_list =  List[Point3D]
    pt3d_list = [Point3D(0, 0, 0) for _ in range(4)]
    pt3d_list[0] = (Point3D(0, 0, 0))
    pt3d_list[1] = (Point3D(0, 1, 0))
    pt3d_list[2] = (Point3D(1, 0, 0))
    pt3d_list[3] = (Point3D(1, 1, 0))

    # Compute the transfer matrix using the approximate_matrix function
    pert1 = 1.0
    pert2 = 0.1
    n = 1000
    for i in range (100):
       matrix = approximate_matrix(matrix, pt3d_list, pt2d_list, width, height, n, pert1)
    return matrix


def line_reverse_project_2d_3d(pt2d_list: List[Point2D], width: int, height: int, inv_matrix: Matrix4x4):
    pt3d_list = []
    for pt2d in pt2d_list:
        pt3d = Point3D()
        pt3d.x, pt3d.y, pt3d.z = project_2d_3d(pt2d, width, height, inv_matrix)
        pt3d_list.append(pt3d)

    return pt3d_list

def calc_rot_angle(pt_center: Point2D, pt_top: Point2D, pt_act: Point2D):
    # Calculate the angle between two points with respect to a center point
    dx1 = pt_act.x - pt_center.x
    dy1 = pt_act.y - pt_center.y
    dx2 = pt_top.x - pt_center.x
    dy2 = pt_top.y - pt_center.y

    #todo: check if this equals 180 - ArcCos((v1.X*v2.X+v1.Y*v2.Y) / (Sqrt(Sqr(v1.X)+Sqr(v1.Y))*Sqrt(Sqr(v2.X)+Sqr(v2.Y)) ))* 180 /pi

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)

    return (angle2 - angle1)


