import unittest
import ltl_diff.ltldiff as ltd
import ltl_diff.constraints as cs
import torch
import numpy as np

class TestConstraints(unittest.TestCase):
    def setUp(self):
        pass

    def test_lipchitz_nonContinuous_posCost(self):
        lc = cs.LipchitzContinuous(10, 0.1)

        ins = torch.ones(1,2,1)
        zs = torch.ones(1,2,1) - 0.1
        targets = None
        net = lambda x: x if (x >= 1).all() else x - 100
        rollout_func = lambda s1, s2, x: [x.expand(1, -1, 1)]

        ltd_const = lc.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreater(loss, 0.0)

    def test_lipchitz_continuous_zeroCost(self):
        lc = cs.LipchitzContinuous(10, 0.1)

        ins = torch.ones(1,2,1)
        zs = torch.ones(1,2,1) - 0.1
        targets = None
        net = lambda x: x[:, 1]
        rollout_func = lambda s1, s2, x: [x.expand(1, 100, 1)]

        ltd_const = lc.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    def test_stayInZone_inZone_zeroCost(self):
        siz = cs.StayInZone(torch.ones(1) * 2.0, torch.ones(1) * 5.0, 0.1)

        ins = None
        zs = torch.ones(1,2,1) * 3.0
        targets = None
        net = lambda x: x[:, 1]
        rollout_func = lambda s1, s2, x: [x.expand(1, 100, 1)]

        ltd_const = siz.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    def test_stayInZone_outZoneLow_PositiveCost(self):
        siz = cs.StayInZone(torch.ones(1) * 2.0, torch.ones(1) * 5.0, 0.1)

        ins = None
        zs = torch.ones(1,2,1) * 1.0
        targets = None
        net = lambda x: x[:, 1]
        rollout_func = lambda s1, s2, x: [x.expand(1, 100, 1)]

        ltd_const = siz.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreater(loss, 0.0)

    def test_stayInZone_outZoneHigh_PositiveCost(self):
        siz = cs.StayInZone(torch.ones(1) * 2.0, torch.ones(1) * 5.0, 0.1)

        ins = None
        zs = torch.ones(1,2,1) * 10.0
        targets = None
        net = lambda x: x[:, 1]
        rollout_func = lambda s1, s2, x: [x.expand(1, 100, 1)]

        ltd_const = siz.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreater(loss, 0.0)

    def test_moveSlowly_slowly_zeroCost(self):
        ms = cs.MoveSlowly(5, 0.1)

        ins = None
        zs = torch.ones(1,2,1) * 4
        targets = None
        net = lambda x: x[:, 1]
        rollout_func = lambda s1, s2, x: [torch.arange(0, 100, step=x.item()).reshape(1,-1,1)]

        ltd_const = ms.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    def test_moveSlowly_fast_positiveCost(self):
        ms = cs.MoveSlowly(5, 0.1)

        ins = None
        zs = torch.ones(1,2,1) * 6
        targets = None
        net = lambda x: x[:, 1]
        rollout_func = lambda s1, s2, x: [torch.arange(0, 100, step=x.item()).reshape(1,-1,1)]

        ltd_const = ms.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreater(loss, 0.0)

    def test_eventuallyReach_pointOnRollout_zeroCost(self):
        er = cs.EventuallyReach([2], 0.1)
        ins = None
        zs = torch.tensor([
            [[1.0], [10.0], [4.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [x[:, er.reach_ids].expand(1, 20, 1)]

        ltd_const = er.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    def test_eventuallyReach_multipleClosePoints_costIsClosestDistance(self):
        er = cs.EventuallyReach([2], 0.1)
        ins = None
        zs = torch.tensor([
            [[1.0], [10.0], [4.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [torch.tensor([3.0, 6.0, 7.0]).reshape(1, 3, 1)]

        ltd_const = er.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreaterEqual(loss, 1.0)
        self.assertLessEqual(loss, 2.0)

    def test_eventuallyReach_manyReachIds_allReached_zeroCost(self):
        er = cs.EventuallyReach([1, 2], 0.1)
        ins = None
        zs = torch.tensor([
            [[1.0], [4.0], [7.0], [10.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [torch.tensor([2.0, 4.0, 7.0]).reshape(1, 3, 1)]

        ltd_const = er.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    def test_eventuallyReach_manyReach_ids_close_closestDistCost(self):
        er = cs.EventuallyReach([1, 2], 0.1)
        ins = None
        zs = torch.tensor([
            [[1.0], [4.0], [7.0], [10.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [torch.tensor([2.0, 10.0]).reshape(1, 2, 1)]

        ltd_const = er.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertAlmostEqual(loss, 3.0, 4)

    # Don't Tip Early (this would be a good one to benefit from the STL time-robustness thing...)
    def test_dontTipEarly_rotationEarly_costIsRotationDiff(self):
        fixed_orientation = 0.0
        dontTip = cs.DontTipEarly(fixed_orientation, 2, 1.0, 0.1)
        ins = None
        zs =  torch.tensor([
            [[1.0, 1.0], [10.0, 10.0], [9.0, 9.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x: [torch.tensor([[
            [1.0, 1.0, 0.0],
            [5.0, 5.0, 0.7],
            [9.0, 9.0, 1.2]
        ]])]

        ltd_const = dontTip.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreater(loss, 0.69)
        self.assertLess(loss, 0.71)

    def test_dontTipEarly_rotationFixedAlways_costZero(self):
        fixed_orientation = 0.0
        dontTip = cs.DontTipEarly(fixed_orientation, 2, 1.0, 0.1)
        ins = None
        zs =  torch.tensor([
            [[1.0, 1.0], [10.0, 10.0], [9.0, 9.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x: [torch.tensor([[
            [1.0, 1.0, 0.0],
            [5.0, 5.0, 0.0],
            [9.0, 9.0, 0.0]
        ]])]

        ltd_const = dontTip.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    def test_dontTipEarly_rotationMovesAfterClose_costZero(self):
        fixed_orientation = 0.0
        dontTip = cs.DontTipEarly(fixed_orientation, 2, 1.0, 0.1)
        ins = None
        zs =  torch.tensor([
            [[1.0, 1.0], [10.0, 10.0], [9.0, 9.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x: [torch.tensor([[
            [1.0, 1.0, 0.0],
            [5.0, 5.0, 0.0],
            [9.0, 9.0, 0.5]
        ]])]

        ltd_const = dontTip.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)

    # Avoid point
    def test_avoidPoint_avoidsPoint_zeroCost(self):
        ap = cs.AvoidPoint(2, 1.0, 0.1)
        ins = None
        zs =  torch.tensor([
            [[1.0], [10.0], [9.0]]
        ])

        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [torch.tensor([[
            [1.0],
            [5.0],
            [6.0]
        ]])]

        ltd_const = ap.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertEqual(loss, 0.0)


    def test_avoidPoint_justOnThePoint_positiveCost(self):
        ap = cs.AvoidPoint(2, 1.0, 0.1)
        ins = None
        zs =  torch.tensor([
            [[1.0], [10.0], [9.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [torch.tensor([[
            [1.0],
            [5.0],
            [8.0]
        ]])]

        ltd_const = ap.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertAlmostEqual(loss, 1.0, places=1)

    def test_avoidPoint_entersPointThenLeaves_stillPositiveCost(self):
        ap = cs.AvoidPoint(2, 1.0, 0.1)
        ins = None
        zs =  torch.tensor([
            [[1.0], [10.0], [9.0]]
        ])
        targets = None
        net = lambda x : x
        rollout_func = lambda s1, s2, x : [torch.tensor([[
            [8.5],
            [8.3],
            [7.2]
        ]])]

        ltd_const = ap.condition(zs, ins, targets, net, rollout_func)
        loss = ltd_const.loss(0).item()
        self.assertGreater(loss, 0.0)