"""
Tests for functions in class SolveDiffusion2D
"""

import numpy as np
import unittest
from unittest import TestCase
from diffusion2d import SolveDiffusion2D

def test_initialize_domain():
    """
    Check function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    
    _w = 25.
    _h = 50.
    _dx = 0.5
    _dy = 0.7
    
    solver.initialize_domain(w=_w, h=_h, dx=_dx, dy=_dy)
    
    assert solver.nx == 50
    assert solver.ny == 71


def test_initialize_physical_parameters():
    """
    Checks function SolveDiffusion2D.initialize_domain
    """
    solver = SolveDiffusion2D()
    
    _d = 5.
    _T_cold = 500.
    _T_hot = 600.
    
    solver.dx = 0.1
    solver.dy = 0.1
    
    solver.initialize_physical_parameters(d=_d, T_cold=_T_cold, T_hot=_T_hot)
    
    assert solver.dt == 0.0005000000000000001


def test_set_initial_condition():
    """
    Checks function SolveDiffusion2D.get_initial_function
    """
    solver = SolveDiffusion2D()
    
    solver.T_cold = 20.
    solver.nx = 5
    solver.ny = 5
    solver.dx = 0.5
    solver.dy = 0.5
    
    _output = solver.set_initial_condition()
    _expected = solver.T_cold * np.ones((solver.nx, solver.ny))

    assert np.equal(_output, _expected).all()
    
class TestDiffusion2D(TestCase):
    def test_initialize_domain(self):
        """
        Check function SolveDiffusion2D.initialize_domain
        """
        solver = SolveDiffusion2D()

        _w = 25.
        _h = 50.
        _dx = 0.5
        _dy = 0.7

        solver.initialize_domain(w=_w, h=_h, dx=_dx, dy=_dy)

        self.assertEqual(solver.nx, 50, "nx is not correct")
        self.assertEqual(solver.ny, 71, "ny is not correct")
    
    def test_initialize_physical_parameters(self):
        """
        Checks function SolveDiffusion2D.initialize_domain
        """
        solver = SolveDiffusion2D()

        _d = 5.
        _T_cold = 500.
        _T_hot = 600.

        solver.dx = 0.1
        solver.dy = 0.1

        solver.initialize_physical_parameters(d=_d, T_cold=_T_cold, T_hot=_T_hot)

        self.assertAlmostEqual(solver.dt,0.0005,delta=0.0000001,msg='dt not correct')
    
    def test_set_initial_condition(self):
        """
        Checks function SolveDiffusion2D.get_initial_function
        """
        solver = SolveDiffusion2D()

        solver.T_cold = 20.
        solver.nx = 5
        solver.ny = 5
        solver.dx = 0.5
        solver.dy = 0.5

        _output = solver.set_initial_condition()
        _expected = solver.T_cold * np.ones((solver.nx, solver.ny))

        self.assertEqual(np.equal(_output, _expected).all(),True,msg='Initial condition not working')

if __name__ == '__main__':
    unittest.main()        