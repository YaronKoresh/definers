from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from definers.media.web_transfer import NetworkTransferStrategy


@dataclass(frozen=True, slots=True)
class HttpTransferCapabilities:
    protocol_preference: str
    http_range_requests: bool
    parallel_connections: bool
    separate_process_workers: bool
    http2_multiplexing: bool
    http2_runtime_ready: bool
    http3_multiplexing: bool
    http3_runtime_ready: bool
    quic_udp: bool
    max_parallel_connections: int
    max_process_workers: int
    max_multiplexed_streams: int


@dataclass(frozen=True, slots=True)
class HttpTransferPolicy:
    runtime_class: str
    protocol_preference: str
    base_strategy_name: str
    strategy_names: tuple[str, ...]
    max_parallel_connections: int
    max_process_workers: int
    max_multiplexed_streams: int


def http_transfer_capabilities() -> HttpTransferCapabilities:
    from definers.media import web_transfer as web_transfer_module

    max_parallel_connections = web_transfer_module._parallel_download_workers()
    max_process_workers = min(
        max_parallel_connections,
        web_transfer_module._download_process_workers(),
    )
    max_multiplexed_streams = min(
        max_parallel_connections,
        web_transfer_module._download_max_multiplexed_streams(),
    )
    return HttpTransferCapabilities(
        protocol_preference=web_transfer_module._download_http_protocol(),
        http_range_requests=True,
        parallel_connections=max_parallel_connections > 1,
        separate_process_workers=max_process_workers > 1,
        http2_multiplexing=web_transfer_module._download_enable_multiplexing(),
        http2_runtime_ready=web_transfer_module._http2_runtime_ready(),
        http3_multiplexing=(
            web_transfer_module._download_enable_multiplexing()
            and web_transfer_module._download_enable_http3()
        ),
        http3_runtime_ready=web_transfer_module._http3_runtime_ready(),
        quic_udp=(
            web_transfer_module._download_enable_multiplexing()
            and web_transfer_module._download_enable_http3()
            and web_transfer_module._http3_runtime_ready()
        ),
        max_parallel_connections=max_parallel_connections,
        max_process_workers=max_process_workers,
        max_multiplexed_streams=max_multiplexed_streams,
    )


def _http_transfer_runtime_class(
    capabilities: HttpTransferCapabilities,
) -> str:
    if not capabilities.parallel_connections:
        return "serial"
    if not capabilities.separate_process_workers:
        return "restricted"
    if (
        capabilities.max_parallel_connections >= 64
        or capabilities.max_multiplexed_streams >= 32
    ):
        return "high-throughput"
    return "standard"


def _http_transfer_base_strategy_name(
    capabilities: HttpTransferCapabilities,
) -> str:
    if (
        capabilities.separate_process_workers
        and capabilities.max_process_workers > 1
    ):
        return "http1-range-process"
    return "http1-range-threaded"


def _http_transfer_protocol_candidates(
    capabilities: HttpTransferCapabilities,
) -> tuple[str, ...]:
    if capabilities.protocol_preference == "http1":
        return ()
    if capabilities.protocol_preference == "http3":
        return ("http3-quic", "http2-multiplex")
    if capabilities.protocol_preference == "http2":
        return ("http2-multiplex", "http3-quic")
    return ("http2-multiplex", "http3-quic")


def http_transfer_policy(
    capabilities: HttpTransferCapabilities | None = None,
) -> HttpTransferPolicy:
    from definers.media import web_transfer as web_transfer_module

    resolved_capabilities = (
        web_transfer_module.http_transfer_capabilities()
        if capabilities is None
        else capabilities
    )
    ordered_strategy_names: list[str] = []
    for strategy_name in _http_transfer_protocol_candidates(
        resolved_capabilities
    ):
        if strategy_name == "http2-multiplex":
            if not (
                resolved_capabilities.http2_multiplexing
                and resolved_capabilities.http2_runtime_ready
            ):
                continue
        elif strategy_name == "http3-quic":
            if not (
                resolved_capabilities.http3_multiplexing
                and resolved_capabilities.http3_runtime_ready
                and resolved_capabilities.quic_udp
            ):
                continue
        ordered_strategy_names.append(strategy_name)
    base_strategy_name = _http_transfer_base_strategy_name(
        resolved_capabilities
    )
    ordered_strategy_names.append(base_strategy_name)
    return HttpTransferPolicy(
        runtime_class=_http_transfer_runtime_class(resolved_capabilities),
        protocol_preference=resolved_capabilities.protocol_preference,
        base_strategy_name=base_strategy_name,
        strategy_names=tuple(ordered_strategy_names),
        max_parallel_connections=resolved_capabilities.max_parallel_connections,
        max_process_workers=resolved_capabilities.max_process_workers,
        max_multiplexed_streams=resolved_capabilities.max_multiplexed_streams,
    )


def _create_http_transfer_strategy_by_name(
    strategy_name: str,
) -> NetworkTransferStrategy:
    from definers.media import web_transfer as web_transfer_module

    if strategy_name == "http1-range-threaded":
        return web_transfer_module.ParallelHttpRangeTransferStrategy()
    if strategy_name == "http1-range-process":
        return web_transfer_module.ParallelProcessHttpRangeTransferStrategy()
    if strategy_name == "http2-multiplex":
        return web_transfer_module.Http2MultiplexedRangeTransferStrategy()
    if strategy_name == "http3-quic":
        return web_transfer_module.Http3MultiplexedRangeTransferStrategy()
    raise LookupError(f"unknown http transfer strategy {strategy_name}")


def create_http_transfer_strategy() -> NetworkTransferStrategy:
    from definers.media import web_transfer as web_transfer_module

    policy = http_transfer_policy()
    strategies = [
        _create_http_transfer_strategy_by_name(strategy_name)
        for strategy_name in policy.strategy_names
    ]
    if len(strategies) == 1:
        return strategies[0]
    return web_transfer_module.AdaptiveHttpTransferStrategy(strategies)
