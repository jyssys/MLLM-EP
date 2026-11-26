"""Map original multimodal placeholder positions onto resident EC rows."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


SLICE_KEY = "campaign_slice_aware_ec_v1"


def suffix_rows(position, prefix_tokens):
    """Return the original embedding row interval consumed after the cut."""
    start = min(position.length, max(0, prefix_tokens - position.offset))
    return position.get_embeds_indices_in_range(start, position.length)


@dataclass(frozen=True)
class SlicedEncoderRows:
    """The compact clone behind V1's logical row-indexed EC lookup.

    The vLLM V1 gather path only reads explicit forward row slices. It gets a
    regular tensor view in return, so modality tagging and model input merging
    retain their existing behavior.
    """

    tensor: "torch.Tensor"
    first_row: int

    def __getitem__(self, rows):
        if not isinstance(rows, slice) or rows.step not in (None, 1):
            raise TypeError("Encoder gather requires a contiguous embedding row slice")
        last_row = self.first_row + self.tensor.shape[0]
        if (
            type(rows.start) is not int
            or type(rows.stop) is not int
            or not self.first_row <= rows.start <= rows.stop <= last_row
        ):
            raise IndexError(
                f"Encoder gather requested {rows} outside resident rows "
                f"[{self.first_row}, {last_row})"
            )
        return self.tensor[rows.start - self.first_row : rows.stop - self.first_row]
