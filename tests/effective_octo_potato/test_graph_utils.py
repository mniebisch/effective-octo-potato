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


class TestMapEdgeIndicesToTemporalGraph:
    def test_two_nodes(self) -> None:
        edge_index = torch.tensor([[0], [1]], dtype=torch.float32)
        num_nodes = 2
        num_frames = 3

        expected = torch.tensor(
            [
                [[0], [1]],
                [[2], [3]],
                [[4], [5]],
            ],
            dtype=torch.float32,
        )

        output = graph_utils.map_edge_indices_to_temporal_graph(
            edge_index=edge_index, num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)

    def test_three_nodes_but_one_edge(self) -> None:
        edge_index = torch.tensor([[0], [2]], dtype=torch.float32)
        num_nodes = 3
        num_frames = 3

        expected = torch.tensor(
            [
                [[0], [2]],
                [[3], [5]],
                [[6], [8]],
            ],
            dtype=torch.float32,
        )

        output = graph_utils.map_edge_indices_to_temporal_graph(
            edge_index=edge_index, num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)

    def test_four_nodes_but_one_edge(self) -> None:
        edge_index = torch.tensor([[0], [2]], dtype=torch.float32)
        num_nodes = 4
        num_frames = 3

        expected = torch.tensor(
            [
                [[0], [2]],
                [[4], [6]],
                [[8], [10]],
            ],
            dtype=torch.float32,
        )

        output = graph_utils.map_edge_indices_to_temporal_graph(
            edge_index=edge_index, num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)


class TestCreateTemporalEdgeIndices:
    def test_simple_case(self) -> None:
        num_nodes = 2
        num_frames = 2

        expected = torch.tensor(
            [
                [0, 1, 2, 3],
                [2, 3, 4, 5],
            ],
            dtype=torch.float32,
        )
        output = graph_utils.create_temporal_edge_indices(
            num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)

    def test_one_node(self) -> None:
        num_nodes = 1
        num_frames = 3

        expected = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
            ],
            dtype=torch.float32,
        )
        output = graph_utils.create_temporal_edge_indices(
            num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)

    def test_correct_frame_handling(self) -> None:
        num_nodes = 2
        num_frames = 4

        expected = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.float32,
        )
        output = graph_utils.create_temporal_edge_indices(
            num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)

    def test_three_nodes(self) -> None:
        num_nodes = 3
        num_frames = 3

        expected = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                [3, 4, 5, 6, 7, 8, 9, 10, 11],
            ],
            dtype=torch.float32,
        )
        output = graph_utils.create_temporal_edge_indices(
            num_nodes=num_nodes, num_frames=num_frames
        )
        torch.testing.assert_close(expected, output)
