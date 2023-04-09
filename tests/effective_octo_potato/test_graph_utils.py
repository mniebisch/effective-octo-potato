import torch

from effective_octo_potato import graph_utils


class TestApplyNodeMaskToEdges:
    def test_three_nodes(self) -> None:
        input_mask = torch.tensor([True, False, True])
        input_edge_index = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [2, 1]]).T

        expected = torch.tensor([[1, 0], [0, 1]])

        output = graph_utils.apply_node_mask_to_edges(input_mask, input_edge_index)

        torch.testing.assert_close(expected, output)

    def test_no_removal(self) -> None:
        input_mask = torch.tensor([True, True, True])
        input_edge_index = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [2, 1]]).T

        expected = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [2, 1]]).T

        output = graph_utils.apply_node_mask_to_edges(input_mask, input_edge_index)

        torch.testing.assert_close(expected, output)

    def test_isolated_removal(self) -> None:
        input_mask = torch.tensor([True, False, True, True])
        input_edge_index = torch.tensor([[0, 2], [2, 3], [3, 0], [0, 3], [3, 2]]).T

        expected = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [2, 1]]).T

        output = graph_utils.apply_node_mask_to_edges(input_mask, input_edge_index)

        torch.testing.assert_close(expected, output)
