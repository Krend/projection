A fun project where I do all the math for 2D->3D perspective projections.

Goal:

Calculate the 3D projection matrix, then project a texture (pixel by pixel) onto the background as if it was in 3D space.

How to use:

* Run ui.py
* Press: 'Calculate'
* Click and drag the mouse on the background image

Based on: 

https://stackoverflow.com/questions/76134/how-do-i-reverse-project-2d-points-into-3d

https://math.stackexchange.com/questions/296794/finding-the-transform-matrix-from-4-projected-points-with-javascript

todo:

* Add tests.
* Add @jit optimization (numba).
* Remove unnecessary types.
* Throw exceptions instead of weird return values.
* Handle these exceptions in the UI.
* Figure out why the image relative coordinates get messed up if I resize the window.
* Add functinality to define a custom quad for the calculation of the projection matrix.