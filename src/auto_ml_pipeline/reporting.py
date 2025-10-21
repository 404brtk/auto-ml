import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available.")

from jinja2 import Environment, PackageLoader, select_autoescape

from auto_ml_pipeline.config import TaskType, ReportConfig
from auto_ml_pipeline.logging_utils import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def generate_report(
        self,
        best_estimator: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        task: TaskType,
        best_model_name: str,
        best_params: Dict,
        cv_score: float,
        test_metrics: Dict,
        scorer_name: str,
        training_time: float,
        all_model_scores: List[Dict[str, Any]],
        report_config: ReportConfig,
        n_splits: int,
        random_state: Optional[int],
    ) -> Path:
        sections = {
            "summary": self._generate_summary(
                best_model_name, cv_score, test_metrics, task, training_time
            ),
            "model_comparison": self._generate_model_comparison_data(all_model_scores),
            "model_performance": self._generate_performance_plots(
                y_test,
                y_pred,
                y_proba,
                task,
                best_estimator.classes_
                if hasattr(best_estimator, "classes_")
                else None,
            ),
            "feature_importance": {},
            "explainability": {},
            "learning_curves": {},
            "hyperparameters": self._prepare_hyperparameters(
                best_params, best_model_name
            ),
            "data_info": self._generate_data_info(X_train, X_test),
        }

        if report_config.include_permutation_importance:
            sections["feature_importance"] = self._generate_feature_importance(
                best_estimator,
                X_train,
                y_train,
                task,
                scorer_name,
                report_config,
                random_state,
            )

        if report_config.include_shap:
            sections["explainability"] = self._generate_explainability_analysis(
                best_estimator, X_train, X_test, report_config, random_state
            )

        if report_config.include_learning_curves:
            sections["learning_curves"] = self._generate_learning_curves(
                best_estimator,
                X_train,
                y_train,
                task,
                scorer_name,
                report_config,
                n_splits,
                random_state,
            )

        html_path = self._create_html_report(sections, report_config)
        logger.info(f"HTML report generated: {html_path}")

        return html_path

    def _to_html(self, fig: go.Figure) -> str:
        """Convert a plotly figure to an HTML div string."""
        return fig.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            default_height="500px",
            default_width="100%",
        )

    def _generate_summary(
        self,
        model_name: str,
        cv_score: float,
        test_metrics: Dict,
        task: TaskType,
        training_time: Optional[float],
    ) -> Dict:
        return {
            "model_name": model_name,
            "task": task.value,
            "cv_score": f"{cv_score:.4f}",
            "test_metrics": test_metrics,
            "training_time": f"{training_time:.2f}s",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _generate_performance_plots(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray],
        task: TaskType,
        classes: Optional[np.ndarray] = None,
    ) -> List[str]:
        plots = []
        if task == TaskType.classification:
            plots.append(self._plot_confusion_matrix(y_test, y_pred, classes))
            if y_proba is not None:
                plots.append(self._plot_roc_curve(y_test, y_proba, classes))
                plots.append(
                    self._plot_precision_recall_curve(y_test, y_proba, classes)
                )
        else:  # regression
            plots.append(self._plot_actual_vs_predicted(y_test, y_pred))
            plots.append(self._plot_residuals(y_test, y_pred))
        return plots

    def _plot_confusion_matrix(
        self,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> str:
        cm = confusion_matrix(y_test, y_pred)
        if classes is None:
            classes = np.unique(y_test)
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Predicted Label", y="True Label", color="Count"),
            x=classes,
            y=classes,
            color_continuous_scale=px.colors.sequential.Blues,
        )
        fig.update_layout(title_text="Confusion Matrix", title_x=0.5)
        return self._to_html(fig)

    def _plot_roc_curve(
        self,
        y_test: np.ndarray,
        y_proba: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> str:
        fig = go.Figure()
        fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

        if y_proba.shape[1] == 2:  # binary
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            auc_score = auc(fpr, tpr)
            name = f"ROC Curve (AUC={auc_score:.3f})"
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))
        else:  # multiclass
            if classes is None:
                classes = np.arange(y_proba.shape[1])
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

            for i in range(y_proba.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                auc_score = auc(fpr, tpr)
                name = f"Class {classes[i]} (AUC={auc_score:.3f})"
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

        fig.update_layout(
            title_text="Receiver Operating Characteristic (ROC) Curve",
            title_x=0.5,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return self._to_html(fig)

    def _plot_precision_recall_curve(
        self,
        y_test: np.ndarray,
        y_proba: np.ndarray,
        classes: Optional[np.ndarray] = None,
    ) -> str:
        fig = go.Figure()

        if y_proba.shape[1] == 2:  # binary
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            fig.add_trace(
                go.Scatter(
                    x=recall, y=precision, name="Precision-Recall Curve", mode="lines"
                )
            )
        else:  # multiclass
            if classes is None:
                classes = np.arange(y_proba.shape[1])
            y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

            for i in range(y_proba.shape[1]):
                precision, recall, _ = precision_recall_curve(
                    y_test_bin[:, i], y_proba[:, i]
                )
                fig.add_trace(
                    go.Scatter(
                        x=recall, y=precision, name=f"Class {classes[i]}", mode="lines"
                    )
                )

        fig.update_layout(
            title_text="Precision-Recall Curve",
            title_x=0.5,
            xaxis_title="Recall",
            yaxis_title="Precision",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return self._to_html(fig)

    def _plot_actual_vs_predicted(self, y_test: np.ndarray, y_pred: np.ndarray) -> str:
        df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        fig = px.scatter(
            df,
            x="Actual",
            y="Predicted",
            opacity=0.7,
            trendline="ols",
            trendline_color_override="red",
            title="Actual vs. Predicted Values",
        )
        fig.update_layout(title_x=0.5)
        return self._to_html(fig)

    def _plot_residuals(self, y_test: np.ndarray, y_pred: np.ndarray) -> str:
        residuals = y_test - y_pred
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Residuals vs. Predicted", "Residuals Distribution"),
        )

        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode="markers", marker=dict(opacity=0.7)),
            row=1,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        fig.add_trace(go.Histogram(x=residuals, nbinsx=30), row=1, col=2)

        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_layout(title_text="Residual Analysis", title_x=0.5, showlegend=False)
        return self._to_html(fig)

    def _generate_feature_importance(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task: TaskType,
        scorer_name: str,
        report_config: ReportConfig,
        random_state: Optional[int],
    ) -> Dict:
        logger.info("Calculating feature importance...")
        plots = []
        feature_names = self._get_feature_names(estimator, X_train)

        builtin_plot = self._plot_builtin_feature_importance(estimator, feature_names)
        if builtin_plot:
            plots.append(builtin_plot)

        perm_plot = self._plot_permutation_importance(
            estimator,
            X_train,
            y_train,
            feature_names,
            task,
            scorer_name,
            report_config,
            random_state,
        )
        if perm_plot:
            plots.append(perm_plot)

        return {"plots": plots, "n_features": len(feature_names)}

    def _get_feature_names(self, estimator: Any, X_train: pd.DataFrame) -> List[str]:
        try:
            # method 1: from preprocessor
            if (
                hasattr(estimator, "named_steps")
                and "preprocessor" in estimator.named_steps
            ):
                preprocessor = estimator.named_steps["preprocessor"]
                if hasattr(preprocessor, "get_feature_names_out"):
                    names = preprocessor.get_feature_names_out()
                    return list(names)

            # method 2: from estimator's stored names
            if hasattr(estimator, "feature_names_in_"):
                names = estimator.feature_names_in_
                return list(names)

            # method 3: from df columns
            if hasattr(X_train, "columns"):
                names = X_train.columns.tolist()
                return names

        except (AttributeError, KeyError, ValueError) as e:
            logger.error(f"Could not extract feature names: {e}")

        raise ValueError("Could not determine feature names from any available method.")

    def _plot_builtin_feature_importance(
        self, estimator: Any, feature_names: List[str]
    ) -> Optional[str]:
        try:
            model = estimator.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                if len(importances) != len(feature_names):
                    logger.warning(
                        f"Length mismatch: {len(importances)} importances vs "
                        f"{len(feature_names)} feature names. Skipping built-in importance."
                    )
                    return None

                df = (
                    pd.DataFrame({"importance": importances, "feature": feature_names})
                    .sort_values(by="importance", ascending=True)
                    .tail(20)
                )
                fig = px.bar(
                    df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Built-in Feature Importance (Top 20)",
                )
                fig.update_layout(title_x=0.5)
                return self._to_html(fig)
        except Exception as e:
            logger.warning(f"Built-in feature importance not available: {e}")
        return None

    def _plot_permutation_importance(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        feature_names: List[str],
        task: TaskType,
        scorer_name: str,
        report_config: ReportConfig,
        random_state: Optional[int],
    ) -> Optional[str]:
        try:
            logger.info("Computing permutation importance...")
            result = permutation_importance(
                estimator,
                X_train,
                y_train,
                n_repeats=report_config.permutation_importance_n_repeats,
                random_state=random_state,
                n_jobs=-1,
                scoring=scorer_name,
            )
            sorted_idx = result.importances_mean.argsort()[-20:]

            fig = go.Figure()
            for i in sorted_idx:
                fig.add_trace(go.Box(x=result.importances[i], name=feature_names[i]))

            fig.update_layout(
                title_text="Permutation Feature Importance (Top 20)",
                title_x=0.5,
                yaxis_title="Feature",
            )
            fig.update_traces(orientation="h")
            return self._to_html(fig)
        except Exception as e:
            logger.warning(f"Permutation importance calculation failed: {e}")
        return None

    def _generate_explainability_analysis(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        report_config: ReportConfig,
        random_state: Optional[int],
    ) -> Dict:
        plots = []
        if SHAP_AVAILABLE:
            shap_plots = self._generate_shap_analysis(
                estimator, X_train, X_test, report_config, random_state
            )
            if shap_plots:
                plots.extend(shap_plots)
        return {"plots": plots, "methods": ["SHAP" if SHAP_AVAILABLE else "N/A"]}

    def _generate_shap_analysis(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        report_config: ReportConfig,
        random_state: Optional[int],
    ) -> List[str]:
        try:
            logger.info("Computing SHAP values...")
            sample_size = min(report_config.shap_sample_size, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=random_state)
            X_test_sample = X_test.sample(
                n=min(sample_size, len(X_test)), random_state=random_state
            )

            preprocessor = estimator.named_steps["preprocessor"]
            model = estimator.named_steps["model"]

            X_sample_transformed = preprocessor.transform(X_sample)
            X_test_transformed = preprocessor.transform(X_test_sample)

            feature_names = self._get_feature_names(estimator, X_train)

            X_sample_transformed_df = pd.DataFrame(
                X_sample_transformed, columns=feature_names
            )
            X_test_transformed_df = pd.DataFrame(
                X_test_transformed, columns=feature_names
            )

            explainer = shap.Explainer(model, X_sample_transformed_df)
            shap_values = explainer(X_test_transformed_df)

            plots = []
            plots.append(
                self._plot_shap_summary_bar(shap_values, X_test_transformed_df)
            )
            plots.append(
                self._plot_shap_summary_beeswarm(shap_values, X_test_transformed_df)
            )
            return plots
        except Exception as e:
            logger.warning(f"SHAP analysis failed: {e}")
        return []

    def _plot_shap_summary_bar(self, shap_values, feature_df) -> str:
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        df = (
            pd.DataFrame(
                {"feature": feature_df.columns, "mean_abs_shap": mean_abs_shap}
            )
            .sort_values("mean_abs_shap", ascending=True)
            .tail(20)
        )
        fig = px.bar(
            df,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="SHAP Mean Absolute Values (Top 20)",
            labels={
                "mean_abs_shap": "mean |SHAP value| (average impact on model output magnitude)"
            },
        )
        fig.update_layout(title_x=0.5)
        return self._to_html(fig)

    def _plot_shap_summary_beeswarm(self, shap_values, feature_df) -> str:
        shaps_df_list = []
        for i, feature in enumerate(feature_df.columns):
            shaps_df_list.append(
                pd.DataFrame(
                    {
                        "feature": feature,
                        "shap_value": shap_values.values[:, i],
                        "feature_value": feature_df.iloc[:, i],
                    }
                )
            )
        shaps_df = pd.concat(shaps_df_list, axis=0)

        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_features = feature_df.columns[np.argsort(mean_abs_shap)[-20:]]
        shaps_df_top = shaps_df[shaps_df["feature"].isin(top_features)]

        fig = px.scatter(
            shaps_df_top,
            x="shap_value",
            y="feature",
            color="feature_value",
            color_continuous_scale="Spectral",
            title="SHAP Summary Plot (Top 20)",
            labels={
                "shap_value": "SHAP value (impact on model output)",
                "feature_value": "Feature Value",
            },
        )
        fig.update_traces(marker=dict(size=4, opacity=0.7))
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
        fig.update_layout(
            title_x=0.5, coloraxis_colorbar=dict(yanchor="top", y=1, ticks="")
        )
        return self._to_html(fig)

    def _generate_learning_curves(
        self,
        estimator: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        task: TaskType,
        scorer_name: str,
        report_config: ReportConfig,
        n_splits: int,
        random_state: Optional[int],
    ) -> Dict:
        try:
            logger.info("Computing learning curves...")
            train_sizes, train_scores, val_scores = learning_curve(
                estimator,
                X_train,
                y_train,
                train_sizes=np.linspace(0.1, 1.0, report_config.learning_curve_steps),
                cv=n_splits,
                scoring=scorer_name,
                n_jobs=-1,
                random_state=random_state,
            )

            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=train_mean,
                    name="Training score",
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=train_mean - train_std,
                    fill=None,
                    mode="lines",
                    line_color="rgba(0,100,80,0.2)",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=train_mean + train_std,
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(0,100,80,0.2)",
                    name="Train score std. dev.",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=val_mean,
                    name="Validation score",
                    mode="lines+markers",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=val_mean - val_std,
                    fill=None,
                    mode="lines",
                    line_color="rgba(255,127,14,0.2)",
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=val_mean + val_std,
                    fill="tonexty",
                    mode="lines",
                    line_color="rgba(255,127,14,0.2)",
                    name="Validation score std. dev.",
                )
            )

            fig.update_layout(
                title_text="Learning Curves",
                title_x=0.5,
                xaxis_title="Training Set Size",
                yaxis_title="Score",
            )
            return {"plot": self._to_html(fig)}
        except Exception as e:
            logger.warning(f"Learning curves generation failed: {e}")
        return {}

    def _prepare_hyperparameters(
        self, params: Dict, model_name: Optional[str] = None
    ) -> Dict:
        return {"model_name": model_name, "params": params if params else {}}

    def _generate_data_info(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
    ) -> Dict:
        return {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": X_train.shape[1],
            "train_test_ratio": f"{len(X_train)}:{len(X_test)}",
        }

    def _generate_model_comparison_data(
        self, all_model_scores: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not all_model_scores:
            return []

        return sorted(all_model_scores, key=lambda x: x["cv_score"], reverse=True)

    def _create_html_report(self, sections: Dict, report_config: ReportConfig) -> Path:
        """Create HTML report using Jinja2 templating."""
        env = Environment(
            loader=PackageLoader("auto_ml_pipeline", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        template = env.get_template("report_template.html")

        feature_importance_config = {
            "enabled": report_config.include_permutation_importance,
            "plots": sections["feature_importance"].get("plots", []),
        }

        explainability_config = {
            "enabled": report_config.include_shap,
            "plots": sections["explainability"].get("plots", []),
            "shap_available": SHAP_AVAILABLE,
        }

        learning_curves_config = {
            "enabled": report_config.include_learning_curves,
            "plot": sections["learning_curves"].get("plot", ""),
        }

        html_content = template.render(
            model_name=sections["summary"]["model_name"],
            timestamp=sections["summary"]["timestamp"],
            task=sections["summary"]["task"],
            cv_score=sections["summary"]["cv_score"],
            training_time=sections["summary"]["training_time"],
            model_scores=sections["model_comparison"],
            test_metrics=sections["summary"]["test_metrics"],
            performance_plots=sections["model_performance"],
            feature_importance=feature_importance_config,
            explainability=explainability_config,
            learning_curves=learning_curves_config,
            hyperparameters=sections["hyperparameters"],
            train_samples=sections["data_info"]["train_samples"],
            test_samples=sections["data_info"]["test_samples"],
            n_features=sections["data_info"].get("n_features", "N/A"),
            train_test_ratio=sections["data_info"]["train_test_ratio"],
        )

        html_path = self.output_dir / "training_report.html"
        html_path.write_text(html_content, encoding="utf-8")
        return html_path
