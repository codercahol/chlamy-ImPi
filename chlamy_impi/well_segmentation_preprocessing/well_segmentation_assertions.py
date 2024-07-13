# In this file we encode our expectations about the shapes of the well segmentations

def assert_expected_shape(i_vals: list, j_vals: list, plate_num: int) -> None:
    if plate_num == 24 or plate_num == 98:  # These plates have two entire missing rows off the bottom
        assert len(i_vals) == 15
        assert len(j_vals) == 25
    else:
        assert len(i_vals) == 17
        assert len(j_vals) == 25
