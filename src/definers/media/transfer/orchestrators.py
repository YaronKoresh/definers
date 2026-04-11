from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from definers.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenException,
    ExponentialBackoffDelay,
    RetryPolicy,
    execute_with_resilience_async,
)

if TYPE_CHECKING:
    from definers.media.web_transfer import NetworkTransferStrategy


@dataclass(frozen=True, slots=True)
class TransferExecutionPolicy:
    max_retries: int = 3
    base_delay_seconds: float = 0.5

    def retry_policy(self) -> RetryPolicy:
        return RetryPolicy(
            max_retries=self.max_retries,
            delay_strategy=ExponentialBackoffDelay(
                base_delay=self.base_delay_seconds
            ),
        )


class ResourceRetrievalOrchestrator:
    def __init__(
        self,
        strategy: NetworkTransferStrategy,
        circuit_breaker: CircuitBreaker | None = None,
        max_retries: int = 3,
        base_delay_seconds: float = 0.5,
    ):
        self.strategy = strategy
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            failure_threshold=3, recovery_timeout=30
        )
        self.execution_policy = TransferExecutionPolicy(
            max_retries=max_retries,
            base_delay_seconds=base_delay_seconds,
        )

    def _log_retry(
        self,
        attempt_number: int,
        total_attempts: int,
        error: BaseException,
    ) -> None:
        logging.getLogger(__name__).warning(
            "Retry attempt %d/%d failed: %s",
            attempt_number,
            total_attempts,
            error,
        )

    async def process(self, source_uri: str, target_node: str | Path) -> bool:
        target_path_object = Path(target_node)

        try:
            return await execute_with_resilience_async(
                self.strategy.execute_transfer,
                source_uri,
                target_path_object,
                circuit_breaker=self.circuit_breaker,
                retry_policy=self.execution_policy.retry_policy(),
                on_retry=self._log_retry,
            )
        except CircuitBreakerOpenException as circuit_open_fault:
            logging.getLogger(__name__).error(
                "Transfer blocked by open circuit: %s", str(circuit_open_fault)
            )
            return False
        except Exception as execution_fault:
            logging.getLogger(__name__).error(
                "Transfer fault: %s", str(execution_fault)
            )
            return False


def create_http_orchestrator() -> ResourceRetrievalOrchestrator:
    from definers.media import web_transfer as web_transfer_module

    return ResourceRetrievalOrchestrator(
        web_transfer_module.create_http_transfer_strategy()
    )


def create_zip_orchestrator() -> ResourceRetrievalOrchestrator:
    from definers.media import web_transfer as web_transfer_module
    from definers.media.web_transfer import ZipExtractTransferStrategy

    return ResourceRetrievalOrchestrator(
        ZipExtractTransferStrategy(
            download_strategy=web_transfer_module.create_http_transfer_strategy()
        )
    )
