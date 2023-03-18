import numpy as np
import torch
from numpy import typing as npt


class FeatureGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: npt.NDArray[np.float32]) -> torch.Tensor:
        """Feature Transformation.

        Args:
            x: Sign sequence with shape:
               [num frames, num landmarks, num spatial dims]
               [num frames, 543, 3].


        Returns:
            Feature vector representing sign sequence with shape [].
            Eg. num landmarks == 85 if only left hand, right hand, and pose
            are used.

        """
        input_shape = x.shape
        if not len(input_shape) == 3 and not input_shape[2] == 3:
            raise ValueError("Invalid input shape.")
        x = torch.from_numpy(x)
        x = self._select_landmarks(x)
        features = self._compute_features(x)
        return self._shape_features(features)

    def _compute_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_features = torch.nanmean(x, dim=0)
        median_features = torch.nanmedian(x, dim=0)
        return torch.nan_to_num(mean_features), torch.nan_to_num(median_features)

    def _select_landmarks(self, x: torch.Tensor) -> torch.Tensor:
        left_hand = torch.arange(458, 489, dtype=torch.long)
        right_hand = torch.arange(522, 543, dtype=torch.long)
        pose = torch.arange(489, 522, dtype=torch.long)
        landmark_indices = torch.cat([left_hand, right_hand, pose])
        return x[:, landmark_indices, :]

    def _shape_features(
        self, features: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        features = torch.concatenate(features, dim=1)
        return torch.flatten(features)


if __name__ == "__main__":
    x_dummy = np.random.rand(20, 543, 3).astype(np.float32)
    fg = FeatureGenerator()
    feat_dummy = fg(x_dummy)
