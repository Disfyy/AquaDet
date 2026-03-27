from ai_core.inference.tracker import SimpleIoUTracker


def test_tracker_persists_id_on_overlap() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3)

    frame1 = tracker.update([(10, 10, 50, 50)])
    frame2 = tracker.update([(12, 12, 52, 52)])

    assert len(frame1) == 1
    assert len(frame2) == 1
    assert frame1[0].track_id == frame2[0].track_id


def test_tracker_creates_new_id_for_far_box() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3)

    frame1 = tracker.update([(10, 10, 50, 50)])
    frame2 = tracker.update([(200, 200, 260, 260)])

    assert frame1[0].track_id != frame2[0].track_id
