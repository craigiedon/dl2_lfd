import unittest
import ltl_diff.ltldiff as ltd
import torch
import numpy as np

class TestLTLDiff(unittest.TestCase):
    def setUp(self):
        pass


    # Equality Tests
    def test_eq_false(self):
        a = ltd.TermStatic(torch.tensor(1.0).reshape(1,1))
        b = ltd.TermStatic(torch.tensor(2.0).reshape(1,1))

        con = ltd.EQ(a, b)
        self.assertEqual(con.satisfy(0), False)
        self.assertEqual(con.loss(0), torch.tensor(1.0))
        self.assertEqual(con.loss(0).shape, torch.Size([1]))

    def test_eq_true(self):
        a = ltd.TermStatic(torch.tensor(2.0).reshape(1,1))
        b = ltd.TermStatic(torch.tensor(2.0).reshape(1,1))

        con = ltd.EQ(a, b)
        self.assertEqual(con.satisfy(0), True)
        self.assertEqual(con.loss(0), torch.tensor(0.0))
        self.assertEqual(con.loss(0).shape, torch.Size([1]))

    def test_eq_dynamic_batch_shape(self):
        a = ltd.TermDynamic(torch.randn(32, 100, 2))
        b = ltd.TermDynamic(torch.randn(32, 100, 2))

        con = ltd.EQ(a,b)
        self.assertEqual(con.loss(0).shape, torch.Size([32]))

    # Greater Than Tests
    def test_gt_batch_shape(self):
        a = ltd.TermDynamic(torch.randn(32, 100, 2))
        b = ltd.TermDynamic(torch.randn(32, 100, 2))

        con = ltd.GT(a, b)
        self.assertEqual(con.loss(0).shape, torch.Size([32]))
        self.assertEqual(con.satisfy(0).shape, torch.Size([32]))

    def test_gt_greater(self):
        a = ltd.TermStatic(torch.tensor([[3.0]]))
        b = ltd.TermStatic(torch.tensor([[2.0]]))

        con = ltd.GT(a, b)
        self.assertEqual(con.loss(0), 0.0)
        self.assertEqual(con.satisfy(0), True)

    def test_gt_less(self):
        a = ltd.TermStatic(torch.tensor([[2.0]]))
        b = ltd.TermStatic(torch.tensor([[3.0]]))

        con = ltd.GT(a, b)
        self.assertEqual(con.loss(0), 1.0)
        self.assertEqual(con.satisfy(0), False)

    def test_gt_equality(self):
        a = ltd.TermStatic(torch.tensor([[2.0]]))
        b = ltd.TermStatic(torch.tensor([[2.0]]))

        con = ltd.GT(a, b)
        self.assertEqual(con.loss(0), 1.0)
        self.assertEqual(con.satisfy(0), False)

    
    # Always Tests
    def test_always_batch_shape(self):
        a = ltd.TermDynamic(torch.randn(32, 100, 2))
        b = ltd.TermDynamic(torch.randn(32, 100, 2))

        con = ltd.Always(ltd.EQ(a, b), 100)
        self.assertEqual(con.loss(0).shape, torch.Size([32]))
        self.assertEqual(con.satisfy(0).shape, torch.Size([32]))

    def test_always_single_batch_shape(self):
        a = ltd.TermDynamic(torch.randn(1, 100, 2))
        b = ltd.TermDynamic(torch.randn(1, 100, 2))

        con = ltd.Always(ltd.EQ(a, b), 100)
        self.assertEqual(con.loss(0).shape, torch.Size([1]))
        self.assertEqual(con.satisfy(0).shape, torch.Size([1]))

    
    def test_always_unequal_stress(self):
        a = ltd.TermDynamic(torch.ones(64, 10000, 7))
        b = ltd.TermDynamic(torch.ones(64, 10000, 7) + 1)

        con = ltd.Always(ltd.EQ(a, b), 10000)

        expected_loss = torch.ones(64) * np.sqrt(7)
        actual_loss = con.loss(0)

        # Note that the numerical errors start to mount up
        # In the end we can only get it to about 1 decimal place of accuracy
        self.assertAlmostEqual(actual_loss[0].item(), expected_loss[0].item(), places=4)
        self.assertEqual(con.satisfy(0).all(), False)

    def test_always_unequal_huge(self):
        a = ltd.TermDynamic(torch.ones(64, 10000, 7))
        b = ltd.TermDynamic(torch.ones(64, 10000, 7) + 1000)

        con = ltd.Always(ltd.EQ(a, b), 10000)

        expected_loss = torch.ones(64) * np.sqrt(7000000)
        actual_loss = con.loss(0)

        self.assertAlmostEqual(actual_loss[0].item(), expected_loss[0].item(), places=1)
        self.assertEqual(con.satisfy(0).all(), False)


    def test_always_equal(self):
        a = ltd.TermDynamic(torch.ones(64, 10000, 7))
        b = ltd.TermDynamic(torch.ones(64, 10000, 7))

        con = ltd.Always(ltd.EQ(a, b), 10000)
        actual_loss = con.loss(0)

        self.assertEqual(actual_loss[0], 0.0)
        self.assertEqual(con.satisfy(0).all(), True)

    # Eventually Tests (Try to break it numerically here)
    def test_eventually_shape(self):
        a = ltd.TermDynamic(torch.ones(64, 10000, 7))
        b = ltd.TermDynamic(torch.ones(64, 10000, 7))

        con = ltd.Eventually(ltd.EQ(a, b), 10000)

        self.assertEqual(con.loss(0).shape, torch.Size([64]))
        self.assertEqual(con.satisfy(0).shape, torch.Size([64]))

    def test_eventually_true_is_zero(self):
        a = ltd.TermDynamic(torch.ones(32, 10, 2))
        b = torch.ones(32, 10, 2) + 1
        b[0][5] -= 1
        b = ltd.TermDynamic(b)

        con = ltd.Eventually(ltd.EQ(a, b), 10)

        self.assertEqual(con.loss(0)[0], 0.0)
        self.assertEqual(con.satisfy(0)[0], True)

    def test_eventually_false_is_nonzero(self):
        a = ltd.TermDynamic(torch.ones(32, 10, 2))
        b = ltd.TermDynamic(torch.ones(32, 10, 2) + 1)

        con = ltd.Eventually(ltd.EQ(a, b), 10)

        self.assertAlmostEqual(con.loss(0)[0].item(), np.sqrt(2.0), places=3)
        self.assertEqual(con.satisfy(0)[0], False)

    def test_eventually_false_stressfull_large(self):
        a = ltd.TermDynamic(torch.ones(32, 100, 7))
        b = ltd.TermDynamic(torch.ones(32, 100, 7) + 100)

        con = ltd.Eventually(ltd.EQ(a, b), 100)

        self.assertAlmostEqual(con.loss(0)[0].item(), np.sqrt(70000), places=3)
        self.assertEqual(con.satisfy(0)[0], False)

    def test_eventually_false_stressful_small(self):
        a = ltd.TermDynamic(torch.ones(32, 100, 1))
        b = ltd.TermDynamic(torch.ones(32, 100, 1) + 0.1)

        con = ltd.Eventually(ltd.EQ(a, b), 100)
        actual_loss = con.loss(0)

        self.assertNotEqual(con.loss(0)[0], 0.0)
        self.assertAlmostEqual(con.loss(0)[0].item(), 0.1)
        self.assertEqual(con.satisfy(0)[0], False)


if __name__ == '__main__':
    unittest.main()
