# src/api/models.py

from pydantic import BaseModel, Field
from typing import List


class Transaction(BaseModel):
    """
    Request schema for Credit Card Fraud Detection.

    Expected feature order (IMPORTANT – must match training):
    V1, V2, ..., V28, norm_amount, norm_time
    Total features = 30
    """

    features: List[float] = Field(
        ...,
        min_items=30,
        max_items=30,
        example=[
            -1.359807, 1.191857, -1.358354, -0.072781,
            2.536347, 1.378155, 0.090794, -0.551600,
            -0.617801, -0.991390, -0.311169, 1.468177,
            -0.470401, 0.207971, 0.624501, 1.106268,
            -0.347201, -0.151152, 0.117316, -0.141208,
            -0.146502, 0.502292, 0.219422, 0.215153,
            0.361610, -0.018307, 0.277838, -0.110474,
            -0.123456, -0.654321
        ]
    )

    class Config:
        schema_extra = {
            "description": "Transaction features after preprocessing (PCA features + scaled amount & time)"
        }
