"""Route actual CPU tensors to a fake fixed forecast owner, without CUDA or weights."""

from unittest.mock import Mock

import pytest

torch = pytest.importorskip('torch')
from vtsfm.adapters.timesfm25.graph_config import FORECAST_GRAPH_SHAPE  # noqa: E402
from vtsfm.adapters.timesfm25.graph_serving import ServingFixedForecast  # noqa: E402


def test_ragged_and_unmatched_shapes_fall_back_before_a_fixed_replay():
    rows, width, horizon = FORECAST_GRAPH_SHAPE
    router = ServingFixedForecast('dense-graph')
    owner = Mock(replay_count=0)
    router.owner = owner
    # Matching non-contiguous inputs must reach the owner as contiguous rows.
    context = torch.arange(width * rows, dtype=torch.float32).reshape(width, rows).T
    ragged = context.clone()
    ragged[0, 0] = torch.nan

    for history, length in ((context[:1], horizon), (context[:, :-1], horizon),
                            (context, horizon - 1), (ragged, horizon)):
        assert router.try_forecast(history, length) is None
    owner.run.assert_not_called()
    assert (router.fallbacks, router.calls, router.graph_replays) == (4, 0, 0)

    forecast = object()

    def replay(history, *, prevalidated):
        assert history.is_contiguous() and prevalidated
        torch.testing.assert_close(history, context)
        owner.replay_count += 1
        return forecast

    owner.run.side_effect = replay
    assert router.try_forecast(context, horizon) is forecast
    owner.run.assert_called_once()
    assert (router.fallbacks, router.calls, router.graph_replays) == (4, 1, 1)


def test_matching_shape_requires_a_warmed_owner():
    rows, width, horizon = FORECAST_GRAPH_SHAPE
    router = ServingFixedForecast('dense-graph')
    with pytest.raises(RuntimeError, match='owner was not initialized'):
        router.try_forecast(torch.ones(rows, width), horizon)
    assert (router.fallbacks, router.calls, router.graph_replays) == (0, 0, 0)
