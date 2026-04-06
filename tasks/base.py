from abc import ABC, abstractmethod

from env.models import AuditReport


class BaseTask(ABC):
    @abstractmethod
    def get_description(self) -> str:
        ...

    @abstractmethod
    def get_table_names(self) -> list[str]:
        ...

    @abstractmethod
    def grade(self, report: AuditReport, gold: dict) -> tuple[float, dict]:
        ...

    @staticmethod
    def brier_adjust(base: float, confidence: float, correct: bool) -> float:
        c = 1.0 if correct else 0.0
        brier = (confidence - c) ** 2
        return base * (1.0 - 0.3 * brier)

    @staticmethod
    def count_accuracy(reported: int, actual: int, tolerance: float = 0.15) -> float:
        if actual == 0:
            return 1.0 if reported == 0 else 0.0
        ratio = abs(reported - actual) / actual
        if ratio <= tolerance:
            return max(0.0, 1.0 - ratio / tolerance)
        return max(0.0, 1.0 - ratio)
