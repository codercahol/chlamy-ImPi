def get_possible_frame_numbers() -> set:
    """The possible numbers of frames in an image array.
    """
    return {84, 92, 100, 164, 172, 180}


def get_time_regime_to_expected_intervals() -> dict[str, set[tuple]]:
    """For each time regime, we have a set of expected time intervals between measurements.
    The intervals are in seconds, found via painful empirical observation.
    """
    return {
        '30s-30s': {(29., 40.), (570, 605), (1760., 1861.)},
        '1min-1min': {(59., 73.), (540, 555), (1730., 1860.)},
        '1min-5min': {(59., 70.), (290., 310.), (420, 433), (540., 555.), (780, 798), (1500, 1522), (1730., 1860.)},
        '5min-5min': {(290., 310.), (1500, 1520), (1730., 1860.)},
        '10min-10min': {(300, 307), (420, 427), (590., 610.), (780, 785), (1500, 1520), (1730., 1860.), (1190., 1220.)},
        '2h-2h': {(1730., 1860.), (599, 603)},
        '20h_ML': {(1730., 1860.), (600, 603)},
        '20h_HL': {(1730., 1860.), (599, 603)}
    }
