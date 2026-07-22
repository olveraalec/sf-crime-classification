from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils.validation import check_is_fitted

from src.logger import get_logger


logger = get_logger(__name__)


CategoricalEncoding = Literal[
    "onehot",
    "ordinal",
    "frequency",
]

NumericStrategy = Literal[
    "standard",
    "robust",
    "passthrough",
    "binned",
]

GeoMode = Literal[
    "raw",
    "cluster",
    "distances",
    "raw_cluster",
    "raw_distances",
    "all",
    "none",
]


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for model-specific fitted preprocessing."""

    categorical_encoding: CategoricalEncoding = "onehot"
    numeric_strategy: NumericStrategy = "standard"

    geo_mode: GeoMode = "raw"
    n_geo_clusters: int = 40

    numeric_bins: int = 10
    sparse_output: bool = True

    random_state: int = 12345
    geo_batch_size: int = 4096


def select_numeric_columns(data: pd.DataFrame) -> list[str]:
    """Select numeric and Boolean columns after geospatial transformation."""
    return data.select_dtypes(include=["number", "bool"]).columns.tolist()


def select_categorical_columns(data: pd.DataFrame) -> list[str]:
    """Select nonnumeric columns after geospatial transformation."""
    numeric = set(select_numeric_columns(data))

    return [column for column in data.columns if column not in numeric]


class AddConstantTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """Add a fixed value to every element."""

    def __init__(self, constant: float = 1.0) -> None:
        self.constant = constant

    def fit(
        self,
        X: np.ndarray,
        y: object = None,
    ) -> AddConstantTransformer:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) + self.constant


class FrequencyEncoder(
    BaseEstimator,
    TransformerMixin,
):
    """
    Encode categories using their training-set relative frequencies.

    Unseen categories receive a frequency of zero. Because the frequency maps
    are learned during fit, this transformer must remain inside the CV pipeline.
    """

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: object = None,
    ) -> FrequencyEncoder:
        frame = self._to_frame(X)

        self.n_features_in_ = frame.shape[1]
        self.frequency_maps_: list[dict[object, float]] = []

        for column in frame.columns:
            frequencies = (
                frame[column]
                .value_counts(
                    normalize=True,
                    dropna=False,
                )
                .to_dict()
            )

            self.frequency_maps_.append(frequencies)

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        check_is_fitted(
            self,
            attributes=[
                "n_features_in_",
                "frequency_maps_",
            ],
        )

        frame = self._to_frame(X)

        if frame.shape[1] != self.n_features_in_:
            raise ValueError(
                "FrequencyEncoder received a different number of "
                "columns during transform."
            )

        encoded_columns = []

        for index, column in enumerate(frame.columns):
            encoded = (
                frame[column]
                .map(self.frequency_maps_[index])
                .fillna(0.0)
                .astype(float)
                .to_numpy()
            )

            encoded_columns.append(encoded)

        return np.column_stack(encoded_columns)

    @staticmethod
    def _to_frame(
        X: pd.DataFrame | np.ndarray,
    ) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True).copy()

        return pd.DataFrame(np.asarray(X))


class GeoSpatialTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    Add fold-safe learned geographic features using MiniBatchKMeans.

    The transformer is fitted only on the training partition supplied by the
    surrounding sklearn Pipeline. Validation and test coordinates are assigned
    using the fitted cluster centers.
    """

    VALID_MODES = {
        "raw",
        "cluster",
        "distances",
        "raw_cluster",
        "raw_distances",
        "all",
        "none",
    }

    def __init__(
        self,
        mode: GeoMode = "raw",
        n_clusters: int = 40,
        longitude_column: str = "longitude",
        latitude_column: str = "latitude",
        random_state: int = 12345,
        batch_size: int = 4096,
    ) -> None:
        self.mode = mode
        self.n_clusters = n_clusters
        self.longitude_column = longitude_column
        self.latitude_column = latitude_column
        self.random_state = random_state
        self.batch_size = batch_size

    def fit(
        self,
        X: pd.DataFrame,
        y: object = None,
    ) -> GeoSpatialTransformer:
        self._validate_input(X)

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown geo mode '{self.mode}'. "
                f"Choose from {sorted(self.VALID_MODES)}."
            )

        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2.")

        coordinates = X[
            [
                self.longitude_column,
                self.latitude_column,
            ]
        ].apply(pd.to_numeric, errors="coerce")

        self.coordinate_medians_ = coordinates.median()

        coordinates = coordinates.fillna(self.coordinate_medians_)

        requires_kmeans = self.mode in {
            "cluster",
            "distances",
            "raw_cluster",
            "raw_distances",
            "all",
        }

        if requires_kmeans:
            self.kmeans_ = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                batch_size=self.batch_size,
                n_init=10,
            )

            self.kmeans_.fit(coordinates)

            logger.info(
                "Fitted MiniBatchKMeans with %s geographic clusters",
                self.n_clusters,
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "coordinate_medians_",
                "feature_names_in_",
            ],
        )

        self._validate_input(X)

        output = X.copy()

        coordinates = output[
            [
                self.longitude_column,
                self.latitude_column,
            ]
        ].apply(pd.to_numeric, errors="coerce")

        coordinates = coordinates.fillna(self.coordinate_medians_)

        add_cluster = self.mode in {
            "cluster",
            "raw_cluster",
            "all",
        }

        add_distances = self.mode in {
            "distances",
            "raw_distances",
            "all",
        }

        retain_raw = self.mode in {
            "raw",
            "raw_cluster",
            "raw_distances",
            "all",
        }

        if add_cluster:
            check_is_fitted(self, attributes=["kmeans_"])

            cluster_labels = self.kmeans_.predict(coordinates)

            # Treat cluster as categorical rather than numeric magnitude.
            output["geo_cluster"] = pd.Series(
                cluster_labels,
                index=output.index,
            ).astype("string")

        if add_distances:
            check_is_fitted(self, attributes=["kmeans_"])

            distance_matrix = self.kmeans_.transform(coordinates)

            for cluster_index in range(self.n_clusters):
                output[f"geo_distance_{cluster_index}"] = distance_matrix[
                    :, cluster_index
                ]

        if not retain_raw:
            output = output.drop(
                columns=[
                    self.longitude_column,
                    self.latitude_column,
                ]
            )

        logger.info(
            "Applied geospatial mode '%s' to %s rows",
            self.mode,
            len(output),
        )

        return output

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("GeoSpatialTransformer requires a pandas DataFrame.")

        missing = {
            self.longitude_column,
            self.latitude_column,
        } - set(X.columns)

        if missing:
            raise KeyError(f"Missing coordinate columns: {sorted(missing)}")


def build_numeric_pipeline(
    strategy: NumericStrategy,
    numeric_bins: int = 10,
    random_state: int = 12345,
) -> Pipeline:
    """Build configurable numeric preprocessing."""

    if strategy == "standard":
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

    if strategy == "robust":
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    RobustScaler(),
                ),
            ]
        )

    if strategy == "passthrough":
        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )

    if strategy == "binned":
        if numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2.")

        return Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "discretizer",
                    KBinsDiscretizer(
                        n_bins=numeric_bins,
                        encode="ordinal",
                        strategy="quantile",
                        subsample=200_000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown numeric strategy: {strategy}")


def build_categorical_pipeline(
    encoding: CategoricalEncoding,
    sparse_output: bool = True,
) -> Pipeline:
    """Build configurable categorical preprocessing."""

    common_imputer = (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="UNKNOWN",
        ),
    )

    if encoding == "onehot":
        return Pipeline(
            steps=[
                common_imputer,
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=sparse_output,
                    ),
                ),
            ]
        )

    if encoding == "ordinal":
        return Pipeline(
            steps=[
                common_imputer,
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )

    if encoding == "frequency":
        return Pipeline(
            steps=[
                common_imputer,
                (
                    "encoder",
                    FrequencyEncoder(),
                ),
            ]
        )

    raise ValueError(f"Unknown categorical encoding: {encoding}")


def build_configurable_transformer(
    config: TransformerConfig,
) -> Pipeline:
    """
    Build a complete fold-safe feature transformation pipeline.

    Geographic KMeans is fit first. Numeric/categorical columns are then
    discovered dynamically from the resulting DataFrame.
    """
    numeric_pipeline = build_numeric_pipeline(
        strategy=config.numeric_strategy,
        numeric_bins=config.numeric_bins,
        random_state=config.random_state,
    )

    categorical_pipeline = build_categorical_pipeline(
        encoding=config.categorical_encoding,
        sparse_output=config.sparse_output,
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                select_numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                select_categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=(
            1.0
            if config.categorical_encoding == "onehot" and config.sparse_output
            else 0.0
        ),
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "geospatial",
                GeoSpatialTransformer(
                    mode=config.geo_mode,
                    n_clusters=config.n_geo_clusters,
                    random_state=config.random_state,
                    batch_size=config.geo_batch_size,
                ),
            ),
            (
                "columns",
                column_transformer,
            ),
        ]
    )


def build_logistic_transformer(
    categorical_encoding: CategoricalEncoding = "onehot",
    numeric_strategy: NumericStrategy = "standard",
    geo_mode: GeoMode = "raw",
    n_geo_clusters: int = 40,
    sparse_output: bool = True,
) -> Pipeline:
    """Build a configurable Logistic Regression transformer."""

    config = TransformerConfig(
        categorical_encoding=categorical_encoding,
        numeric_strategy=numeric_strategy,
        geo_mode=geo_mode,
        n_geo_clusters=n_geo_clusters,
        sparse_output=sparse_output,
    )

    return build_configurable_transformer(config)


def build_tree_transformer(
    categorical_encoding: CategoricalEncoding = "ordinal",
    numeric_strategy: NumericStrategy = "passthrough",
    geo_mode: GeoMode = "raw_cluster",
    n_geo_clusters: int = 40,
) -> Pipeline:
    """Build a configurable transformer for tree models."""

    config = TransformerConfig(
        categorical_encoding=categorical_encoding,
        numeric_strategy=numeric_strategy,
        geo_mode=geo_mode,
        n_geo_clusters=n_geo_clusters,
        sparse_output=False,
    )

    return build_configurable_transformer(config)


def build_naive_bayes_transformer(
    numeric_bins: int = 10,
    geo_mode: GeoMode = "cluster",
    n_geo_clusters: int = 40,
) -> Pipeline:
    """
    Build the discrete, non-negative CategoricalNB transformer.

    Categorical unknown values are shifted from -1 to 0, while known values
    become 1, 2, 3, and so on.
    """
    numeric_pipeline = build_numeric_pipeline(
        strategy="binned",
        numeric_bins=numeric_bins,
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="UNKNOWN",
                ),
            ),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
            (
                "shift",
                AddConstantTransformer(constant=1),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                select_numeric_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                select_categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            (
                "geospatial",
                GeoSpatialTransformer(
                    mode=geo_mode,
                    n_clusters=n_geo_clusters,
                ),
            ),
            (
                "columns",
                column_transformer,
            ),
        ]
    )