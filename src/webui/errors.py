"""User-facing (Ukrainian) error texts for the API.

The frontend shows ``detail`` verbatim, so these strings are part of the HTTP
contract (pinned by tests/test_api_contract.py).
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Error raised by services; the app turns it into ``{"detail": detail}`` with ``status_code``.

    Keeps FastAPI out of the service layer while producing exactly the response
    shape of ``HTTPException`` that the frontend already reads.
    """

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


PIPELINE_VALUE_ERROR_UK = {
    "No person detected in front image": (
        "На знімку анфасу не виявлено людину. Переконайтеся, що фігура повністю в кадрі "
        "та поза відповідає вимогам."
    ),
    "No person detected in side image": (
        "На знімку профілю не виявлено людину. Переконайтеся, що фігура повністю в кадрі "
        "та поза відповідає вимогам."
    ),
    "No body silhouette detected in front image": (
        "На анфасі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "No body silhouette detected in side image": (
        "На профілі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "No segmentation mask for front image": (
        "На анфасі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "No segmentation mask for side image": (
        "На профілі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "Cannot calibrate front view: insufficient visible keypoints": (
        "Недостатньо видимих ключових точок на анфасі для калібровки за зростом. "
        "Переконайтеся, що ступні та голова в кадрі."
    ),
    "Cannot calibrate side view: insufficient visible keypoints": (
        "Недостатньо видимих ключових точок на профілі для калібровки за зростом. "
        "Переконайтеся, що ступні та голова в кадрі."
    ),
    "Invalid sex for measurement pipeline": "Некоректне значення статі для пайплайну.",
}


def pipeline_value_error_detail(message: str) -> str:
    """Ukrainian text for a ValueError raised by the measurement pipeline."""
    return PIPELINE_VALUE_ERROR_UK.get(
        message.strip(),
        f"Не вдалося обробити знімки: {message}",
    )


def validation_errors_to_uk(errors: list[Any]) -> str:
    """Join FastAPI/pydantic request-validation errors into one Ukrainian message."""
    if not errors:
        return "Некоректні дані форми."
    parts: list[str] = []
    field_labels = {
        "height_cm": "Зріст (см)",
        "sex": "Стать",
        "front": "Фото анфасу",
        "side": "Фото профілю",
        "pose_backend": "Модель пози",
    }
    for item in errors:
        if not isinstance(item, dict):
            continue
        loc = tuple(item.get("loc") or ())
        field_key = str(loc[-1]) if loc else "form"
        label = field_labels.get(field_key, field_key)
        err_type = str(item.get("type") or "")
        msg_en = str(item.get("msg") or "")
        ctx = item.get("ctx")
        if not isinstance(ctx, dict):
            ctx = {}

        if err_type == "missing":
            parts.append(f"{label}: значення не передано.")
        elif err_type in ("float_parsing", "decimal_parsing", "int_parsing"):
            parts.append(f"{label}: потрібне число.")
        elif err_type == "greater_than_equal":
            ge = ctx.get("ge")
            parts.append(f"{label}: занадто мале значення (мінімум {ge}).")
        elif err_type == "less_than_equal":
            le = ctx.get("le")
            parts.append(f"{label}: занадто велике значення (максимум {le}).")
        elif err_type in ("literal_error", "enum"):
            parts.append(f"{label}: недопустиме значення.")
        else:
            parts.append(f"{label}: {msg_en}")
    return " ".join(parts) if parts else "Некоректні дані форми."
