from typing import List
"""
This module provides various mathematical functions and utilities for matrix operations, 
3D to 2D projections, and 2D to 3D projections. It includes classes for representing 
points, matrices, and geometric shapes, as well as functions for transformations, 
projections, and approximations.
Classes:
    - Point2D: Represents a 2D point with x and y coordinates.
    - Point3D: Represents a 3D point with x, y, and z coordinates.
    - Matrix3x3: Represents a 3x3 matrix.
    - Matrix4x4: Represents a 4x4 matrix.
    - Quad: Represents a quadrilateral defined by four 2D points.
    - Rect: Represents a rectangle defined by two 2D points.
Functions:
    - adjugate3x3(matrix): Computes the adjugate of a 3x3 matrix.
    - determinant3x3(matrix): Computes the determinant of a 3x3 matrix.
    - inverse3x3(matrix): Computes the inverse of a 3x3 matrix.
    - invers4x4(matrix): Computes the inverse of a 4x4 matrix.
    - multiply3x3(matrix_a, matrix_b): Multiplies two 3x3 matrices.
    - get_coeffs(x1, x2, x3, x4, y1, y2, y3, y4): Computes coefficients for a transformation.
    - source_to_dest(x_source, y_source, combined_matrix): Transforms a 2D point using a 3x3 matrix.
    - quad_to_rect(quad): Computes the bounding rectangle of a quadrilateral.
    - project_3d_2d(pt3d, width, height, matrix): Projects a 3D point onto a 2D plane using a 4x4 matrix.
    - project_2d_3d(pt2d, width, height, inv_matrix): Projects a 2D point onto 3D space using an inverse 4x4 matrix.
    - point_dist_2d(pt1, pt2): Computes the squared distance between two 2D points.
    - eval_dist(matrix, pt3d_list, pt2d_list, width, height): Evaluates the projection error between 3D and 2D points.
    - perturb_matrix(matrix, perturbation): Perturbs a 4x4 matrix by a random value.
    - approximate_matrix(matrix, pt3d_list, pt2d_list, width, height, n, perturbation): Approximates a transformation matrix.
    - compute_transfer_matrix(pt2d_list, width, height): Computes a transfer matrix for 2D points.
    - list_reverse_project_2d_3d(pt2d_list, width, height, inv_matrix): Projects a list of 2D points onto 3D space.
    - list_project_3d_2d(pt3d_list, width, height, matrix): Projects a list of 3D points onto a 2D plane.
    - calc_rot_angle(pt_center, pt_top, pt_act): Calculates the rotation angle between two points relative to a center.
    - line_to_unit_square_3d_3d(pt_start, pt_end): Maps a line segment to a unit square in 3D space.
Notes:
    - The module assumes that input matrices have the correct dimensions (3x3 or 4x4).
    - Functions return -1 or (-1, -1, -1) in cases where a valid result cannot be computed.
    - The `approximate_matrix` function uses random perturbations to iteratively improve the transformation matrix.
"""
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

def adjugate3x3(matrix: Matrix3x3):
    """
    Computes the adjugate (adjoint) of a 3x3 matrix.
    The adjugate of a matrix is the transpose of its cofactor matrix. This function
    assumes the input is a 3x3 matrix represented as a flat list of 9 elements.
    Args:
        matrix (Matrix3x3): A 3x3 matrix represented as an object with a property `m`,
                            which is a flat list of 9 elements in row-major order.
    Returns:
        list: A flat list of 9 elements representing the adjugate of the input matrix
              in row-major order.
    Raises:
        AssertionError: If the input matrix does not contain exactly 9 elements.
    Example:
        Given a 3x3 matrix:
            | a b c |
            | d e f |
            | g h i |
        The adjugate is computed as:
            | ei-fh ch-bi bf-ce |
            | fg-di ai-cg cd-af |
            | dh-eg bg-ah ae-bd |
    """
    assert len(matrix.m) == 9, "Matrix must have 9 elements"
        
    return [matrix.m[4]*matrix.m[8]-matrix.m[5]*matrix.m[7], matrix.m[2]*matrix.m[7]-matrix.m[1]*matrix.m[8], matrix.m[1]*matrix.m[5]-matrix.m[2]*matrix.m[4],
            matrix.m[5]*matrix.m[6]-matrix.m[3]*matrix.m[8], matrix.m[0]*matrix.m[8]-matrix.m[2]*matrix.m[6], matrix.m[2]*matrix.m[3]-matrix.m[0]*matrix.m[5],
            matrix.m[3]*matrix.m[7]-matrix.m[4]*matrix.m[6], matrix.m[1]*matrix.m[6]-matrix.m[0]*matrix.m[7], matrix.m[0]*matrix.m[4]-matrix.m[1]*matrix.m[3] ]

def determinant3x3(matrix: Matrix3x3):
    """
    Calculate the determinant of a 3x3 matrix.

    The determinant is computed using the formula:
        det = a(ei − fh) − b(di − fg) + c(dh − eg)
    where the matrix is represented as:
        | a  b  c |
        | d  e  f |
        | g  h  i |

    Args:
        matrix (Matrix3x3): A 3x3 matrix represented as an object with a 
                            flat list of 9 elements (matrix.m).

    Returns:
        float: The determinant of the 3x3 matrix.

    Raises:
        AssertionError: If the matrix does not contain exactly 9 elements.
    """
    assert len(matrix.m) == 9, "Matrix must have 9 elements"

    return matrix.m[0]*(matrix.m[4]*matrix.m[8]-matrix.m[5]*matrix.m[7]) - matrix.m[1]*(matrix.m[3]*matrix.m[8]-matrix.m[5]*matrix.m[6]) + matrix.m[2]*(matrix.m[3]*matrix.m[7]-matrix.m[4]*matrix.m[6])

def inverse3x3(matrix: Matrix3x3):
    """
    Computes the inverse of a 3x3 matrix.
    Args:
        matrix (Matrix3x3): The input 3x3 matrix to be inverted. It must have exactly 9 elements.
    Returns:
        Matrix3x3: The inverted matrix if the determinant is non-zero.
        int: Returns -1 if the determinant is zero, indicating the matrix is not invertible.
    Raises:
        AssertionError: If the input matrix does not have exactly 9 elements.
    Notes:
        - The function calculates the determinant of the input matrix. If the determinant is zero,
          the matrix is singular and cannot be inverted.
        - The adjugate of the matrix is computed and divided by the determinant to obtain the inverse.
    """
    assert len(matrix.m) == 9, "Matrix must have 9 elements"
    
    inv_matrix = Matrix3x3()
    
    det = determinant3x3(matrix)

    # two rows equal
    if det == 0:    #todo: throwing an exception might be better
        return -1
    
    adj = adjugate3x3(matrix)

    for i in range(9):
        inv_matrix.m[i] = adj[i] / det

    return inv_matrix


def invers4x4(matrix: Matrix4x4):
    """
    Computes the inverse of a 4x4 matrix.
    This function calculates the inverse of a 4x4 matrix represented by the 
    `Matrix4x4` class. The input matrix must have exactly 16 elements stored 
    in a flat list or array-like structure. If the determinant of the matrix 
    is zero, indicating that the matrix is non-invertible, the function 
    returns -1.
    Args:
        matrix (Matrix4x4): The input 4x4 matrix to be inverted. It must have 
                            an attribute `m` which is a list of 16 elements.
    Returns:
        Matrix4x4: The inverted matrix if the determinant is non-zero.
        int: Returns -1 if the matrix is non-invertible (determinant is zero).
    Raises:
        AssertionError: If the input matrix does not have exactly 16 elements.
    Note:
        - The function assumes that the input matrix is stored in row-major 
          order.
        - The determinant is calculated as part of the inversion process, and 
          the inverse is scaled by the reciprocal of the determinant.
    """
    assert len(matrix.m) == 16, "Matrix must be have 16 elements"
    
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

def multiply3x3(matrix_a: Matrix3x3, matrix_b: Matrix3x3): 
    """
    Multiplies two 3x3 matrices and returns the resulting matrix.

    Args:
        matrix_a (Matrix3x3): The first 3x3 matrix, represented as an object with a flat list `m` containing 9 elements.
        matrix_b (Matrix3x3): The second 3x3 matrix, represented as an object with a flat list `m` containing 9 elements.

    Returns:
        Matrix3x3: The resulting 3x3 matrix after multiplication.

    Raises:
        AssertionError: If either input matrix does not contain exactly 9 elements.

    Note:
        The matrices are assumed to be stored in row-major order.
    """
    assert len(matrix_a.m) == 9 and len(matrix_b.m) == 9, "Matrix must have 9 elements"

    matrix_c = Matrix3x3()
    for i in range(3):
        for j in range(3):
            field_sum = 0
            for k in range(3):
                field_sum += (matrix_a.m[i * 3 + k] * matrix_b.m[k * 3 + j])
            matrix_c.m[i * 3 + j] = field_sum    

    return matrix_c

# Advanced projections functions

def get_coeffs(x1: float, x2: float, x3: float, x4: float, y1: float, y2: float, y3: float, y4: float):
    """
    Computes the coefficients (l, g, t) of the following system of liner equations:
        | x1  x2  x3 |   |l|   |x4|
        | y1  y2  y2 | * |g| = |y4|
        |  1   1   1 |   |t|   | 1|

    """
    denom = (x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2)
    
    if denom == 0:
        return -1, -1, -1   #todo: check if this can be a valid result
        
    l = ((x2 * y3 - x3 * y2 + x4 * (y2 - y3) - y4 * (x2 - x3)) / denom)
    g = ((-x1 * y3 + x3 * y1 - x4 * (y1 - y3) + y4 * (x1 - x3)) / denom)
    t = ((x1 * y2 - x2 * y1 + x4 * (y1 - y2) - y4 * (x1 - x2)) / denom)
    
    return l, g, t

# matrix.m size of 9
def source_to_dest(x_source: int, y_source: int, combined_matrix: Matrix3x3):
    assert len(combined_matrix.m) == 9, "Matrix must have 9 elements"

    xTmp = combined_matrix.m[0] * x_source + combined_matrix.m[1] * y_source + combined_matrix.m[2]
    yTmp = combined_matrix.m[3] * x_source + combined_matrix.m[4] * y_source + combined_matrix.m[5]
    zTmp = combined_matrix.m[6] * x_source + combined_matrix.m[7] * y_source + combined_matrix.m[8]

    if zTmp == 0.0:
        return -1, -1   #todo: check if this can be a valid result
       
    return round(xTmp / zTmp), round(yTmp / zTmp)

# Return the rectange which is around the quad
def quad_to_rect(quad: Quad): 
    """
    Converts a quadrilateral (Quad) into a bounding rectangle (Rect).
    This function takes a quadrilateral object with four points and calculates
    the smallest rectangle that can fully enclose the quadrilateral. The rectangle
    is defined by two points: pt0 (top-left corner) and pt1 (bottom-right corner).
    Args:
        quad (Quad): An object representing a quadrilateral, which must have
                     exactly four points (quad.pts) with x and y coordinates.
    Returns:
        Rect: A rectangle object with pt0 and pt1 defining the bounding box.
    Raises:
        AssertionError: If the input quadrilateral does not have exactly four points.
    """
    assert len(quad.pts) == 4, "Quad must have 4 points"
    
    rect = Rect()
    rect.pt0.x = quad.pts[0].x
    rect.pt1.x = quad.pts[0].x
    rect.pt0.y = quad.pts[0].y
    rect.pt1.y = quad.pts[0].y

    for i in range(1, 4):
        rect.pt0.x = min(rect.pt0.x, quad.pts[i].x)
        rect.pt1.x = max(rect.pt1.x, quad.pts[i].x)
        rect.pt0.y = min(rect.pt0.y, quad.pts[i].y)
        rect.pt1.y = max(rect.pt1.y, quad.pts[i].y)

    return rect

# This function is used to project a 3D point onto a 2D plane using a transformation matrix.m.
# It takes a 3D point (Pt), a transformation matrix.m (matrix.m), and the width and height of the 2D plane as inputs.
# The function returns the projected 2D coordinates of the point.
def project_3d_2d(pt3d: Point3D, width: int, height: int, matrix: Matrix4x4):
    """
    Projects a 3D point onto a 2D plane using a 4x4 transformation matrix.
    Args:
        pt3d (Point3D): The 3D point to be projected, represented as an object with x, y, and z attributes.
        width (int): The width of the 2D viewpoert.
        height (int): The height of the 2D viewport.
        matrix (Matrix4x4): A 4x4 transformation matrix represented as an object with a 16-element list `m`.
    Returns:
        tuple: A tuple (x, y) representing the 2D coordinates of the projected point on the plane.
               If the projection is invalid (e.g., w == 0), returns (-1, -1).
    Raises:
        AssertionError: If the provided matrix does not have exactly 16 elements.
    Notes:
        - The z-axis is not used in the 2D projection.
        - The function assumes the matrix is in row-major order.
        - The result is normalized to fit within the dimensions of the 2D plane.
    """
    assert len(matrix.m) == 16, "Matrix must have 16 elements"
    x = matrix.m[0]  * pt3d.x + matrix.m[1]  * pt3d.y + matrix.m[2]  * pt3d.z + matrix.m[3]
    y = matrix.m[4]  * pt3d.x + matrix.m[5]  * pt3d.y + matrix.m[6]  * pt3d.z + matrix.m[7]
    # z axis is not used in 2D projection
    w = matrix.m[12] * pt3d.x + matrix.m[13] * pt3d.y + matrix.m[14] * pt3d.z + matrix.m[15]
    
    if w == 0:
        return -1, -1   #todo: check if this can be a valid result
        
    return (width * (x / w + 1.0) / 2), (height * (y / w + 1.0) / 2)

# This function is used to project a 2D point onto a 3D space using an inverse transformation matrix.m.
# It takes a 2D point (Pt), an inverse transformation matrix.m (inv_matrix.m), and the width and height of the 2D plane as inputs.
# The function returns the projected 3D coordinates of the point.
def project_2d_3d(pt2d: Point2D, width: int, height: int, inv_matrix: Matrix4x4):
    """
    Projects a 2D point into 3D space using an inverse transformation matrix.
    Args:
        pt2d (Point2D): The 2D point to be projected, with attributes `x` and `y`.
        width (int): The width of the 2D viewport, used for normalization.
        height (int): The height of the 2D viewport, used for normalization.
        inv_matrix (Matrix4x4): A 4x4 inverse transformation matrix, represented as 
                                an object with a flat list `m` containing 16 elements.
    Returns:
        tuple: A tuple containing the projected 3D coordinates (x, y, z). If the 
               homogeneous coordinate `w` is zero, returns (-1, -1, -1) as an error indicator.
    Raises:
        AssertionError: If the provided matrix does not contain exactly 16 elements.
    """
    assert len(inv_matrix.m) == 16, "Matrix must have 16 elements"
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
    """
    Calculate the squared distance between two points in 2D space.

    This function computes the squared distance between two points,
    which is the sum of the squared differences of their x and y coordinates.
    The squared distance is used instead of the exact distance to avoid
    the computational cost of a square root operation when only the magnitude
    is needed.

    Args:
        pt1 (Point2D): The first point in 2D space.
        pt2 (Point2D): The second point in 2D space.

    Returns:
        float: The squared distance between the two points.
    """
    dx = pt2.x - pt1.x
    dy = pt2.y - pt1.y
    return dx * dx + dy * dy # not exact distance, we only need the magnitude

def eval_dist(matrix: Matrix4x4, pt3d_list : List[Point3D], pt2d_list: List[Point2D], width: int, height: int):        
    """
    Evaluates the total distance between a list of projected 2D points and a corresponding list of target 2D points.
    This function projects a list of 3D points onto a 2D plane using a given transformation matrix and screen dimensions.
    It then calculates the sum of distances between the projected points and the corresponding target 2D points.
    Args:
        matrix (Matrix4x4): A 4x4 transformation matrix used for projecting 3D points to 2D.
        pt3d_list (List[Point3D]): A list of 3D points to be projected.
        pt2d_list (List[Point2D]): A list of target 2D points to compare against.
        width (int): The width of the 2D projection plane (screen).
        height (int): The height of the 2D projection plane (screen).
    Returns:
        float: The total distance between the projected 2D points and the target 2D points.
    Raises:
        AssertionError: If the matrix does not have 16 elements.
        AssertionError: If the lengths of `pt3d_list` and `pt2d_list` do not match.
        AssertionError: If `width` or `height` is not positive.
    """
    assert len(matrix.m) == 16, "Matrix must have 16 elements"
    assert len(pt3d_list) == len(pt2d_list), "Point lists must have the same length"
    assert width > 0 and height > 0, "Width and height must be positive"
    
    dist = 0
    for i in range(len(pt3d_list)):
        pt2d = Point2D(0, 0)
        pt2d.x, pt2d.y = project_3d_2d(pt3d_list[i], width, height, matrix)
        dist += point_dist_2d(pt2d, pt2d_list[i])

    return dist

def perturb_matrix(matrix: Matrix4x4, perturbation: float):
    """
    Applies a random perturbation to one element of a 4x4 matrix.

    This function takes a 4x4 matrix and modifies one of its elements by adding
    a random value within the range [-perturbation, perturbation]. The matrix
    is represented as a flat list of 16 elements.

    Args:
        matrix (Matrix4x4): The input 4x4 matrix to be perturbed. It must have
            exactly 16 elements stored in the `m` attribute.
        perturbation (float): The maximum magnitude of the random perturbation
            to be applied to a single element of the matrix.

    Returns:
        Matrix4x4: A new 4x4 matrix with one element randomly perturbed.

    Raises:
        AssertionError: If the input matrix does not have exactly 16 elements.

    Note:
        The input matrix is not modified. A new perturbed matrix is returned.
    """
    assert len(matrix.m) == 16, "Matrix must have 16 elements"
    perturbed_matrix = Matrix4x4()
    for i in range(len(matrix.m)):
        perturbed_matrix.m[i] = matrix.m[i]

    i = random.randrange(0, 16)    
    perturbed_matrix.m[i] += random.uniform(-perturbation, perturbation)
    return perturbed_matrix

def approximate_matrix(matrix: Matrix4x4, pt3d_list : List[Point3D], pt2d_list: List[Point2D], width: int, height: int, n:  int, perturbation: float):
    """
    Approximates a transformation matrix by iteratively perturbing it to minimize
    the distance between projected 3D points and their corresponding 2D points.
    Args:
        matrix (Matrix4x4): The initial 4x4 transformation matrix to be optimized.
        pt3d_list (List[Point3D]): A list of 3D points to be projected.
        pt2d_list (List[Point2D]): A list of corresponding 2D points for the 3D points.
        width (int): The width of the projection space (e.g., screen or image width).
        height (int): The height of the projection space (e.g., screen or image height).
        n (int): The number of iterations to perform for optimization.
        perturbation (float): The magnitude of the perturbation applied to the matrix.
    Returns:
        Matrix4x4: The optimized transformation matrix that minimizes the projection error.
    Raises:
        AssertionError: If the input matrix does not have 16 elements.
        AssertionError: If width or height is not positive.
        AssertionError: If the number of iterations (n) is not positive.
        AssertionError: If the lengths of pt3d_list and pt2d_list do not match.
    """
    assert len(matrix.m) == 16, "Matrix must have 16 elements"
    assert width > 0 and height > 0, "Width and height must be positive"
    assert n > 0, "Number of iterations must be positive"
    assert len(pt3d_list) == len(pt2d_list), "Point lists must have the same length"
    
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
    """
    Computes a 4x4 transfer matrix that maps a set of 3D points to their corresponding 2D points
    on a projection plane, given the dimensions of the plane.

    Args:
        pt2d_list (List[Point2D]): A list of 4 2D points representing the target positions on the 
                                   projection plane. The list must contain exactly 4 points.
        width (int): The width of the projection plane. Must be a positive integer.
        height (int): The height of the projection plane. Must be a positive integer.

    Returns:
        Matrix4x4: A 4x4 transformation matrix that best fits the mapping between the 3D points 
                   and the given 2D points.

    Raises:
        AssertionError: If `pt2d_list` does not contain exactly 4 points.
        AssertionError: If `width` or `height` is not a positive integer.

    Notes:
        - The function initializes the transformation matrix as an identity matrix and iteratively 
          refines it using the `approximate_matrix` function.
        - The 3D points are predefined as a unit square in the XY plane at Z=0.
        - Two levels of perturbation (`pert1` and `pert2`) are used to refine the matrix over 100 iterations.
    """
    assert len(pt2d_list) == 4, "Point list must have 4 points"
    assert width > 0 and height > 0, "Width and height must be positive"
    matrix = Matrix4x4()
    # Initialize the matrix with identity values 
    # We will distort this until we get the best fit for the points
    matrix.m[0] = 1.0
    matrix.m[5] = 1.0
    matrix.m[10] = 1.0
    matrix.m[15] = 1.0   

    pt3d_list: List[Point3D]
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
       matrix = approximate_matrix(matrix, pt3d_list, pt2d_list, width, height, n, pert2)
    return matrix


def list_reverse_project_2d_3d(pt2d_list: List[Point2D], width: int, height: int, inv_matrix: Matrix4x4):
    """
    Projects a list of 2D points back into 3D space using an inverse transformation matrix.

    Args:
        pt2d_list (List[Point2D]): A list of 2D points to be projected.
        width (int): The width of the 2D viewport
        height (int): The height of the 2D viewport.
        inv_matrix (Matrix4x4): A 4x4 inverse transformation matrix used for the projection.

    Returns:
        List[Point3D]: A list of 3D points obtained by projecting the input 2D points.

    Raises:
        AssertionError: If the inverse matrix does not have 16 elements.
        AssertionError: If the width or height is not positive.
        AssertionError: If the input point list is empty.
    """
    assert len(inv_matrix.m) == 16, "Matrix must have 16 elements"
    assert width > 0 and height > 0, "Width and height must be positive"
    assert len(pt2d_list) > 0, "Point list must have at least one point"

    pt3d_list: List[Point3D]  = []
    for pt2d in pt2d_list:
        pt3d = Point3D(0, 0, 0)
        pt3d.x, pt3d.y, pt3d.z = project_2d_3d(pt2d, width, height, inv_matrix)
        pt3d_list.append(pt3d)

    return pt3d_list

def list_project_3d_2d(pt3d_list: List[Point3D], width: int, height: int, matrix: Matrix4x4):
    """
    Projects a list of 3D points onto a 2D plane using a given transformation matrix.

    Args:
        pt3d_list (List[Point3D]): A list of 3D points to be projected.
        width (int): The width of the 2D viewport.
        height (int): The height of the 2D viewport.
        matrix (Matrix4x4): A 4x4 transformation matrix used for the projection.

    Returns:
        List[Point2D]: A list of 2D points resulting from the projection.

    Raises:
        AssertionError: If the matrix does not have 16 elements.
        AssertionError: If the width or height is not positive.
        AssertionError: If the input list of 3D points is empty.
    """
    assert len(matrix.m) == 16, "Matrix must have 16 elements"
    assert width > 0 and height > 0, "Width and height must be positive"
    assert len(pt3d_list) > 0, "Point list must have at least one point"

    pt2d_list: List[Point2D]  = []
    for pt3d in pt3d_list:
        pt2d = Point2D(0, 0)
        pt2d.x, pt2d.y = project_3d_2d(pt3d, width, height, matrix)
        pt2d.x = round(pt2d.x)
        pt2d.y = round(pt2d.y)
        pt2d_list.append(pt2d)

    return pt2d_list

def calc_rot_angle(pt_center: Point2D, pt_top: Point2D, pt_act: Point2D):
    """
    Calculate the rotation angle between two points with respect to a center point.
    This function computes the angle formed by the vector from `pt_center` to `pt_act` 
    and the vector from `pt_center` to `pt_top`. The result is adjusted to represent 
    the rotation angle in radians.
    Args:
        pt_center (Point2D): The center point of rotation.
        pt_top (Point2D): The reference point defining the top direction.
        pt_act (Point2D): The active point for which the angle is calculated.
    Returns:
        float: The rotation angle in radians.
    """
    # Calculate the angle between two points with respect to a center point
    dx1 = pt_act.x - pt_center.x
    dy1 = pt_act.y - pt_center.y
    dx2 = pt_top.x - pt_center.x
    dy2 = pt_top.y - pt_center.y

    #todo: check if this equals 180 - ArcCos((v1.X*v2.X+v1.Y*v2.Y) / (Sqrt(Sqr(v1.X)+Sqr(v1.Y))*Sqrt(Sqr(v2.X)+Sqr(v2.Y)) ))* 180 /pi

    angle1 = math.atan2(dy1, dx1)
    angle2 = math.atan2(dy2, dx2)
        
    return math.pi - (angle2 - angle1)
    #return math.pi - math.acos((dx1*dx2+dy1*dy2) / (math.sqrt((dx1*dx1)+(dy1*dy1))*math.sqrt((dx2*dx2)+(dy2*dy2))))

def line_to_unit_square_3d_3d(pt_start: Point2D,  pt_end: Point2D):
    out_rec_3d: List[Point3D]
    out_rec_3d = [Point3D(0, 0, 0) for _ in range(4)]
    out_pt_unit_start = Point3D(0, 0, 0)

    pt_top  = Point2D(pt_start.x, pt_start.y + 1)
    angle_rad = calc_rot_angle(pt_start, pt_top, pt_end)

    out_pt_unit_start.x = -1 * math.sin(angle_rad) + pt_end.x
    out_pt_unit_start.y = math.cos(angle_rad) + pt_end.y

    norm = 0.5
    out_rec_3d[0].x = norm * math.cos(angle_rad) + out_pt_unit_start.x
    out_rec_3d[0].y = norm * math.sin(angle_rad) + out_pt_unit_start.y

    out_rec_3d[1].x = norm  * math.cos(angle_rad) + pt_end.x
    out_rec_3d[1].y = norm  * math.sin(angle_rad) + pt_end.y

    out_rec_3d[2].x = -norm * math.cos(angle_rad) + pt_end.x
    out_rec_3d[2].y = -norm * math.sin(angle_rad) + pt_end.y

    out_rec_3d[3].x = -norm * math.cos(angle_rad) + out_pt_unit_start.x
    out_rec_3d[3].y = -norm * math.sin(angle_rad) + out_pt_unit_start.y

    return out_rec_3d, out_pt_unit_start    
