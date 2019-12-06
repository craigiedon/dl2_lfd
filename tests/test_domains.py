import unittest
import ltl_diff.ltldiff as ltd
from ltl_diff.domains import Box
import torch
import numpy as np

class TestDomains(unittest.TestCase):
    def setUp(self):
        pass

    
    def test_empty_box_empty_raise_error(self):
        x1 = torch.ones(64, 7) * 2
        x2 = torch.ones(64, 7)

        with self.assertRaises(AssertionError):
            b = Box(x1, x2)

    def test_non_empty_box_false(self):
        x1 = torch.ones(64, 7)
        x2 = torch.ones(64, 7) * 2
        b = Box(x1, x2)

        self.assertFalse(b.is_empty())

    def test_projectInside_identity(self):
        x1 = torch.ones(64, 7)
        x2 = torch.ones(64, 7) * 10.0

        b = Box(x1, x2)
        unproj = torch.ones(64, 7) * 5.0
        proj = b.project(unproj)
        self.assertTrue(torch.equal(proj, unproj))

    def test_projectOutsideMax_max(self):
        x1 = torch.ones(64, 7)
        x2 = torch.ones(64, 7) * 10.0

        b = Box(x1, x2)
        unproj = torch.ones(64, 7) * 25
        proj = b.project(unproj)
        self.assertTrue(torch.equal(proj, x2))

    def test_projectOutsideMin_min(self):
        x1 = torch.ones(64, 7)
        x2 = torch.ones(64, 7) * 10.0

        b = Box(x1, x2)
        unproj = torch.ones(64, 7) * -100.0
        proj = b.project(unproj)
        self.assertTrue(torch.equal(proj, x1))

    def test_samplesObeyDomain(self):
        x1 = torch.ones(64, 7)
        x2 = torch.ones(64, 7) * 10.0

        b = Box(x1, x2)
        s = b.sample()
        self.assertTrue(((s >= x1) & (s <= x2)).all())

