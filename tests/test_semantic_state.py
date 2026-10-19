import json

import pandas as pd

from jev_tmcad.semantic_state import QuantileSchema, build_semantic_state


def test_state_excludes_label_and_paths():
    row = pd.Series(
        {
            "sample_id": "abc",
            "label": "gear",
            "split": "test",
            "absolute_path": r"D:\\TMCAD\\gear\\gear_1.stp",
            "relative_path": "gear/gear_1.stp",
            "dim_small_large_ratio": 0.2,
            "dim_mid_large_ratio": 0.9,
            "pca_eig_ratio_1": 0.8,
            "pca_eig_ratio_2": 0.1,
            "bbox_fill_ratio": 0.3,
            "face_count": 50,
            "edge_count": 100,
            "vertex_count": 80,
            "solid_count": 1,
            "shell_count": 1,
            "largest_face_area_fraction": 0.2,
            "face_area_frac_plane": 0.4,
            "face_area_frac_cylinder": 0.5,
            "face_area_frac_cone": 0.0,
            "face_area_frac_torus": 0.0,
            "face_area_frac_bspline": 0.0,
            "edge_length_frac_circle": 0.6,
            "edge_length_frac_line": 0.3,
            "best_rotational_symmetry": 0.9,
            "best_axis_periodicity_strength": 0.5,
            "best_axis_periodicity_order": 12,
            "best_axis_axial_variation": 0.2,
            "best_axis_end_asymmetry": 0.1,
            "best_axis_elongation": 1.2,
            "best_axis_discness": 2.0,
            "best_axis_angular_profile": json.dumps([1.0] * 8),
            "best_axis_axial_profile": json.dumps([0.8] * 6),
        }
    )
    schema = QuantileSchema(
        cuts={
            k: [0.1, 0.2, 0.7, 0.9]
            for k in [
                "face_count",
                "edge_count",
                "vertex_count",
                "solid_count",
                "shell_count",
                "best_rotational_symmetry",
                "best_axis_periodicity_strength",
                "best_axis_axial_variation",
                "best_axis_end_asymmetry",
                "bbox_fill_ratio",
                "surface_to_bbox_surface_ratio",
            ]
        }
    )
    state = build_semantic_state(row, schema)
    text = json.dumps(state).lower()
    assert "gear_1" not in text
    assert "d:\\tmcad" not in text
    assert "sample_id" not in text
    assert "true_label" not in text
    assert '"label"' not in text
