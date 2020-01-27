import unittest
import ltl_diff.oracle as oracle
import ltl_diff.constraints as cs
import ltl_diff.ltldiff as ltd
from ltl_diff.domains import Box
import torch
import numpy as np

class TestOracle(unittest.TestCase):
    def setUp(self):
        self.doubler = lambda x: x * 2

    # Eval simple constraint with no domains. Should skip and go straight to a clear loss
    def test_noDom_fixed_loss(self):
        ins = torch.tensor([-2.0, 0.0, 2.0]).reshape(3, 1 ,1)
        targets = ins * 2
        constraint = OutputMoreDummy(0.1)
        domains = []
        zs = oracle.general_attack(ins, targets, constraint, domains, 100,
                                   self.doubler, None)
        self.assertIsNone(zs)

    # Simple constraint, already satsified. Should have no way to generate a failure case
    def test_satisfied_fail_to_gen(self):
        ins = torch.tensor([1.0, 2.0, 3.0]).reshape(3, 1, 1)
        targets = ins * 2
        constraint = OutputMoreDummy(0.1)
        domains = constraint.domains(ins, targets)
        zs = oracle.general_attack(ins, targets, constraint, domains, 100,
                                   self.doubler, None)

        neg_loss, pos_loss, sat = cs.constraint_loss(constraint, ins, targets, zs, self.doubler, None)

        self.assertTrue((pos_loss == 0.0).all())
        self.assertTrue((neg_loss > 0.0).all())
        self.assertTrue(sat.all())

    # Simple one that is super easy to violate. Should get to some high neg loss
    def test_easy_to_break(self):
        ins = torch.tensor([-1.0, -2.0, -3.0]).reshape(3, 1, 1)
        targets = ins * 2
        constraint = OutputMoreDummy(0.1)
        domains = constraint.domains(ins, targets)
        zs = oracle.general_attack(ins, targets, constraint, domains, 100,
                                   self.doubler, None)

        neg_loss, pos_loss, sat = cs.constraint_loss(constraint, ins, targets, zs, self.doubler, None)

        self.assertTrue((pos_loss > 0.0).all())
        self.assertTrue((neg_loss == 0.0).all())
        self.assertFalse(sat.any())

    def test_mixed_sats(self):
        ins = torch.tensor([-2.0, -1.0, 0.0, 0.05, 0.2]).reshape(-1, 1, 1)
        targets = ins * 2
        constraint = OutputMoreDummy(0.1)
        domains = constraint.domains(ins, targets)
        zs = oracle.general_attack(ins, targets, constraint, domains, 100,
                                   self.doubler, None)

        neg_loss, pos_loss, sat = cs.constraint_loss(constraint, ins, targets, zs, self.doubler, None)

        self.assertTrue((pos_loss[0:4] >= 0.0).all())
        self.assertGreater(neg_loss[4], 0.0)
        self.assertTrue((neg_loss[0:4] == 0.0).all())

        expected_sats = torch.tensor([False, False, False, False, True]).reshape(5, 1)
        self.assertTrue((sat == expected_sats).all())

    # A potentially unbounded domain. How quickly does it get to the max violating one?
    def test_enormous_domain_counterexample_maxes_out(self):
        ins = torch.tensor([10.0]).reshape(1, 1, 1)
        targets = ins * 2
        constraint = OutputMoreDummy(11.0)
        domains = constraint.domains(ins, targets)
        zs = oracle.general_attack(ins, targets, constraint, domains, 100,
                                   self.doubler, None)

        neg_loss, pos_loss, sat = cs.constraint_loss(constraint, ins, targets, zs, self.doubler, None)

        self.assertGreater(pos_loss, 0.0)
        self.assertEqual(neg_loss, 0.0)
        self.assertFalse(sat)


class OutputMoreDummy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        return ltd.LEQ(ltd.TermStatic(zs), ltd.TermStatic(weights))

    def domains(self, ins, targets):
        return cs.fully_global_ins(ins, self.epsilon)
