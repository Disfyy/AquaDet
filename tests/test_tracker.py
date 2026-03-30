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


def test_tracker_velocity_prediction() -> None:
    """After many consistent displacements, the tracker should predict motion
    and still match even if the displacement is large enough to reduce raw IoU."""
    tracker = SimpleIoUTracker(iou_threshold=0.3)

    # Simulate an object moving steadily rightward by 5px per frame
    bbox = (100, 100, 150, 150)
    first = tracker.update([bbox])
    first_id = first[0].track_id

    for i in range(1, 10):
        shifted = (100 + 5 * i, 100, 150 + 5 * i, 150)
        result = tracker.update([shifted])
        assert result[0].track_id == first_id, f"Lost track at frame {i}"


def test_tracker_removes_stale_tracks() -> None:
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_missed=3)

    tracker.update([(10, 10, 50, 50)])
    # Miss the object for 5 consecutive frames
    for _ in range(5):
        tracker.update([])

    # Old track should be expired
    frame = tracker.update([(10, 10, 50, 50)])
    # Should get a new ID since old one was pruned
    assert len(frame) == 1
