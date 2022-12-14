# BEM200

This repository contain a simple implementation of the Acoustical Boundary Element Method. This means that the script contain all code relevant for solving the following boundary integral equation in 2d.

$$
c(\mathbf{y})p(\mathbf{y}) = \int_\Gamma G(\mathbf{x},\mathbf{y})\frac{\partial p}{\partial \mathbf{n}}(\mathbf{x})\ \mathrm{d}\Gamma_\mathbf{x} - \int_\Gamma\frac{\partial G(\mathbf{x}, \mathbf{y})}{\partial \mathbf{n} }p(\mathbf{x})\ \mathrm{d}\Gamma_\mathbf{x} + p_\text{incident}(\mathbf{x}).
$$

The script implements linear and quadratic geometry as well as discontinuous constant, linear and quadratic interpolation of the pressure and normal derivative. 

An infinite cylinder with hard boundaries is used to verify the correctness of the implementation. 
