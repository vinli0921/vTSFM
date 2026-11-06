"""CPU-safe validation at the worker's forecast routing boundary."""


def order_requests(requests, request_ids):
    """Match transport payloads to actual worker slots, never arrival order."""
    requests, request_ids = tuple(requests), tuple(request_ids)
    by_id = {request.request_id: request for request in requests}
    if (not requests or len(by_id) != len(requests)
            or len(set(request_ids)) != len(request_ids)
            or len(request_ids) != len(requests) or set(request_ids) != set(by_id)):
        raise ValueError('Worker request IDs must match transported requests exactly once')
    return tuple(by_id[identity] for identity in request_ids)


def validate_forecast(request, output, quantile_count):
    expected = (len(request.item_ids), quantile_count, request.horizon)
    if tuple(output.shape) != expected:
        raise ValueError(f'Forecast for {request.request_id!r} must have shape {expected}, got {output.shape}')
    if str(output.dtype) != str(request.context.dtype):
        raise ValueError('Forecast dtype must match its numerical context')
    if str(getattr(output, 'device', 'cpu')) != str(getattr(request.context, 'device', 'cpu')):
        raise ValueError('Forecast must remain on its numerical context device')
