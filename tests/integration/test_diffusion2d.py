"""
Tests for functionality checks in class SolveDiffusion2D
"""

from diffusion2d import SolveDiffusion2D
import numpy as np

def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.test_initialize_physical_parameters
    """
    solver = SolveDiffusion2D()
    
    _w = 10. 
    _h = 20.
    _dx = 30.
    _dy = 40.
    
    solver.initialize_domain(_w, _h, _dx, _dy)
    
    _d = 20.
    _T_Cold = 400. 
    _T_Hot = 600.
    
    solver.initialize_physical_parameters(_d, _T_Cold, _T_Hot)
    
    _expeted_dt = 14.4
    
    assert solver.dt == _expeted_dt


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.set_initial_condition
    """
    solver = SolveDiffusion2D()
    
    _w = 20.
    _h =  30.
    _dx = 5.
    _dy = 5.
    
    _T_Cold = 400. 
    _T_Hot = 800.
    
    solver.initialize_domain(_w, _h, _dx, _dy)
    
    solver.T_cold = _T_Cold
    solver.T_hot = _T_Hot

    _output = solver.set_initial_condition()

    _expected = _T_Cold * np.ones((solver.nx, solver.ny))
    
    r, cx, cy = 2, 5, 5  
    r2 = r ** 2

    for i in range(solver.nx):
        for j in range(solver.ny):
            p2 = (i * solver.dx - cx) ** 2 + (j * solver.dy - cy) ** 2
            if p2 < r2:
                _expected[i, j] = _T_Hot
    
    assert np.equal(_output, _expected).all()        
