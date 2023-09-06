Basic Raytracing Effects
========================

For this assignment I implemented basic ray tracing effects such as shadows, reflection, refraction and depth of field.

### Using Eigen

This was done using Eigen
Have a look at the [Getting Started](http://eigen.tuxfamily.org/dox/GettingStarted.html) page of Eigen as well as the [Quick Reference](http://eigen.tuxfamily.org/dox/group__QuickRefPage.html) page for a reference of the basic matrix operations supported.

Ex 0: Implement the intersection code:
------------------------------------------

Filled the functions `ray_sphere_intersection` and `ray_parallelogram_intersection` with the correct intersection between the ray and the primitives.

Output with sphere intersection
![](build/sphere.png)

Output with sphere and plane intersection
![](build/sphere-plane.png)


Ex 1: Field of View and Perspective Camera:
------------------------------------------

![](img/fov.png?raw=true)

Filled the starter code to compute the correct value of `h` (`image_y` in the code).
Implemented the perspective camera similarly to Assignment 1.


Output with correct `image_x` and `image_y`
![](build/fov-res.png)

Output with correct `image_x` and `image_y` and perspective camera (remember to change `is_perspective` to `true`)
![](build/prespective.png)


Ex.2: Shadow Rays:
-----------------


Implemented Phong shading (diffuse and specular color)Fill in Implemented shadow rays by implementing the function `is_light_visible`.

Output with correct shading
![](img/phong.png)

Output with shadows
![](img/shadow.png)


Reflection:
-----------------------

Implemented reflected rays

Output with reflections
![](img/reflections.png)


Perlin Noise
-------------------------

Implemented Perlin noise as explained in class.


Implemented the linear interpolation
Implemented the `dotGridGradient` function
Get the correct grid coordinates from the point `x` and `y`
Replaced the linear interpolation with a cubic interpolation `(a1 - a0) * (3.0 - w * 2.0) * w * w + a0` and compared the results.


Output with linear interpolation
![](img/perlin-lin.png)

Output with cubic interpolation
![](img/perlin-cub.png)