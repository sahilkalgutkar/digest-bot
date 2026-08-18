from datetime import datetime, timedelta, timezone

from retrieve.search import recency_weight


def test_recency_weight_is_almost_one_for_now():
    now = datetime.now(timezone.utc).isoformat()
    assert recency_weight(now, half_life_hours=24) > 0.9999


def test_recency_weight_decreases_with_age():
    now = datetime.now(timezone.utc)
    recent = (now - timedelta(hours=1)).isoformat()
    old = (now - timedelta(hours=100)).isoformat()

    assert recency_weight(recent, half_life_hours=24) > recency_weight(old, half_life_hours=24)


def test_recency_weight_approaches_zero_for_very_old_articles():
    ancient = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
    assert recency_weight(ancient, half_life_hours=24) < 0.001


def test_recency_weight_clamps_future_timestamps_to_full_weight():
    future = (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()
    assert recency_weight(future, half_life_hours=24) == 1.0
