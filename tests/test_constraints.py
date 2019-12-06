import unittest
import ltl_diff.ltldiff as ltd
import torch
import numpy as np

class TestConstraints(unittest.TestCase):
    def setUp(self):
        pass

    # Can get on with testing
    def lipchitz_nonContinuous_posCost(self):
        pass

    def lipchitz_continuous_zeroCost(self):
        pass

    def stayInZone_inZone_zeroCost(self):
        pass

    def stayInZone_outZoneLow_PositiveCost(self):
        pass

    def stayInZone_outZoneHigh_PositiveCost(self):
        pass

    def moveSlowly_slowly_zeroCost(self):
        pass

    def moveSlowly_fast_positiveCost(self):
        pass

    # Needs modification to be based on inputs rather than fixed pre-config...
    def eventuallyReach_pointOnRollout_zeroCost(self):
        pass

    def eventuallyReach_multipleClosePoints_costIsClosestDistance(self):
        pass

    # Don't Tip Early (this would be a good one to benefit from the STL time-robustness thing...)
    def dontTipEarly_rotationEarly_costIsRotationDiff(self):
        pass

    def dontTipEarly_rotationFixedAlways_costZero(self):
        pass

    def dontTipEarly_rotationMovesAfterClose_costZero(self):
        pass

    # Avoid point
    def avoidPoint_avoidsPoint_zeroCost(self):
        pass

    def avoidPoint_justOnThePoint_positiveCost(self):
        pass

    def avoidPoint_entersPointThenLeaves_stillPositiveCost(self):
        pass