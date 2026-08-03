import torch

from calib.collect_stats import (
    TOKEN_TEXT,
    TOKEN_VISION,
    TOKEN_VISION_KEY,
    TOKEN_VISION_REDUNDANT,
    collect_calibration_stats,
    collect_layer_stats,
    collect_layer_stats_from_attention,
    load_sharegpt4v_records,
    resolve_calibration_image_path,
    split_vision_tokens_by_attention,
)


def test_collect_layer_stats_counts_normalization_and_key_red_split():
    expert_assignment = torch.tensor(
        [
            [0, 1],
            [0, 1],
            [1, 2],
            [2, 3],
        ]
    )
    token_type = torch.tensor(
        [
            TOKEN_VISION_KEY,
            TOKEN_TEXT,
            TOKEN_VISION_REDUNDANT,
            TOKEN_TEXT,
        ]
    )
    hidden_states = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
        ]
    )

    stats = collect_layer_stats(expert_assignment, token_type, hidden_states, num_experts=4, delta=0.1)

    assert torch.equal(stats["N_vis"], torch.tensor([1.0, 2.0, 1.0, 0.0]))
    assert torch.equal(stats["N_txt"], torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert torch.isclose(stats["f_vis"].sum(), torch.tensor(1.0))
    assert torch.isclose(stats["f_txt"].sum(), torch.tensor(1.0))
    assert stats["delta"][1] > 0
    assert stats["delta"][3] < 0
    assert torch.isclose(stats["f_key"].sum(), torch.tensor(1.0))
    assert torch.isclose(stats["f_red"].sum(), torch.tensor(1.0))
    assert torch.allclose(stats["centroid"][0], torch.tensor([1.5, 0.0]))


def test_attention_split_matches_mode_adaptive_top20_and_redundant_rest():
    token_type = torch.tensor(
        [
            TOKEN_TEXT,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_TEXT,
            TOKEN_TEXT,
        ]
    )
    input_ids = torch.tensor([10, 151655, 151655, 151655, 151655, 151655, 20, 999])
    attention = torch.zeros(1, 1, 8, 8)
    attention[0, 0, 0, 1] = 100.0
    attention[0, 0, 6, 4] = 5.0
    attention[0, 0, 7, 2] = 10.0

    split = split_vision_tokens_by_attention(
        attention=attention,
        token_type=token_type,
        dominant_ratio=0.2,
        attn_mode="adaptive",
        input_ids=input_ids,
        special_token_ids=[999],
    )

    assert torch.equal(split["key_mask"], torch.tensor([False, False, False, False, True, False, False, False]))
    assert torch.equal(split["redundant_mask"], torch.tensor([False, True, True, True, False, True, False, False]))
    assert split["vision_importance"][4] > split["vision_importance"][2]


def test_collect_layer_stats_from_attention_counts_mode_key_redundant_and_centroid():
    token_type = torch.tensor(
        [
            TOKEN_TEXT,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_VISION,
            TOKEN_TEXT,
        ]
    )
    expert_assignment = torch.tensor([[0], [1], [1], [2], [3], [2], [0]])
    hidden_states = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    attention = torch.zeros(1, 1, 7, 7)
    attention[0, 0, 6, 4] = 7.0

    stats = collect_layer_stats_from_attention(
        expert_assignment=expert_assignment,
        token_type=token_type,
        hidden_states=hidden_states,
        num_experts=4,
        attention=attention,
        dominant_ratio=0.2,
    )

    assert torch.equal(stats["N_key"], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert torch.equal(stats["N_red"], torch.tensor([0.0, 2.0, 2.0, 0.0]))
    assert torch.equal(stats["N_dominant_image"], stats["N_key"])
    assert torch.equal(stats["N_redundant_image"], stats["N_red"])
    assert torch.allclose(stats["centroid"][3], hidden_states[4])


def test_collect_calibration_stats_accepts_attention_payload():
    layers = {
        0: {
            "expert_assignment": torch.tensor([[0], [1], [1]]),
            "token_type": torch.tensor([TOKEN_TEXT, TOKEN_VISION, TOKEN_TEXT]),
            "hidden_states": torch.ones(3, 2),
            "attention": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                ]
            ),
            "dominant_ratio": 0.2,
            "attn_mode": "adaptive",
        }
    }

    stats = collect_calibration_stats(layers, num_experts=2)

    assert torch.equal(stats[0]["N_key"], torch.tensor([0.0, 1.0]))
    assert stats[0]["vision_importance"][1] == 2.0


def test_sharegpt4v_calibration_records_and_image_resolution(tmp_path):
    calib_path = tmp_path / "calib.json"
    calib_path.write_text(
        """
        [
          {
            "id": "sample-0",
            "image": "coco/train2017/000000000009.jpg",
            "conversations": [
              {"from": "human", "value": "<image>\\nWhat is shown?"},
              {"from": "gpt", "value": "A scene."}
            ]
          }
        ]
        """,
        encoding="utf-8",
    )

    records = load_sharegpt4v_records(calib_path)
    image_path = resolve_calibration_image_path(records[0], tmp_path / "data")

    assert records[0]["image"] == "coco/train2017/000000000009.jpg"
    assert image_path == tmp_path / "data" / "coco/train2017/000000000009.jpg"
