import unittest

import torch

from w0.trainer import group_optimizer_parameters


class OptimizerGroupTests(unittest.TestCase):
    def test_grouping_preserves_order_and_unsplit_parameters(self):
        parameters = [torch.nn.Parameter(torch.zeros(n)) for n in (3, 4, 11, 2)]
        self.assertIs(group_optimizer_parameters(parameters), parameters)
        groups = group_optimizer_parameters(parameters, 7)
        self.assertEqual([sum(p.numel() for p in g["params"]) for g in groups], [7, 11, 2])
        self.assertEqual([id(p) for g in groups for p in g["params"]], [id(p) for p in parameters])
        with self.assertRaises(ValueError):
            group_optimizer_parameters(parameters, -1)

    def test_grouping_preserves_adamw_updates(self):
        torch.manual_seed(7)
        plain = [torch.nn.Parameter(torch.randn(n)) for n in (3, 4, 11, 2)]
        grouped = [torch.nn.Parameter(p.detach().clone()) for p in plain]
        optimizers = [
            torch.optim.AdamW(plain, lr=1e-7, weight_decay=0, foreach=False),
            torch.optim.AdamW(group_optimizer_parameters(grouped, 7), lr=1e-7,
                              weight_decay=0, foreach=False),
        ]
        for _ in range(3):
            for a, b in zip(plain, grouped):
                a.grad = torch.randn_like(a)
                b.grad = a.grad.clone()
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        for a, b in zip(plain, grouped):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
