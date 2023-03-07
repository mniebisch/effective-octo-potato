"""Test for data submodule."""

import pandas as pd

from effective_octo_potato.data import create_idx_map


class TestCreateIdxMap:
    def test_one_to_n_map(self) -> None:
        input_data = pd.DataFrame({"frame_min": [1], "frame_max": [3]})
        expected = [(0, 1), (0, 2), (0, 3)]
        output = create_idx_map(data=input_data)

        assert expected == output

    def test_n_to_m_map(self) -> None:
        input_data = pd.DataFrame({"frame_min": [4], "frame_max": [7]})
        expected = [(0, 4), (0, 5), (0, 6), (0, 7)]
        output = create_idx_map(data=input_data)

        assert expected == output

    def test_single_frame_map(self) -> None:
        input_data = pd.DataFrame({"frame_min": [1], "frame_max": [1]})
        expected = [(0, 1)]
        output = create_idx_map(data=input_data)

        assert expected == output

    def test_multiple_input_rows(self) -> None:
        input_data = pd.DataFrame({"frame_min": [1, 2], "frame_max": [1, 4]})
        expected = [(0, 1), (1, 2), (1, 3), (1, 4)]
        output = create_idx_map(data=input_data)

        assert expected == output
