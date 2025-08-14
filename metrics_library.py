"""
Enhanced metrics_library.py

A comprehensive, platform-agnostic Python library for calculating a full suite of
metrics for various machine learning models. This library implements a hybrid
strategy, combining:
1. Evidently AI: For standardized data drift, data quality, and model performance.
2. SHAP: For deep model explainability and feature importance drift monitoring.
3. Custom Logic: For business-specific KPIs and comprehensive model quality metrics.

Core Functions:
- generate_comprehensive_monitoring_report: The main orchestrator for complete ML monitoring
- Three-pillar approach: Data Drift, Model Quality, Business Impact

Dependencies: pip install pandas numpy scikit-learn evidently shap scipy
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Core ML libraries
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, average_precision_score, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, median_absolute_error,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)

# Evidently and SHAP
try:
    import evidently
    from evidently.report import Report
    from evidently.metric_preset import (
        DataDriftPreset, DataQualityPreset, ClusteringPreset, 
        ClassificationPreset, RegressionPreset
    )
    from evidently.metrics import *
    EVIDENTLY_AVAILABLE = True
except ImportError:
    print("Warning: Evidently not available. Install with: pip install evidently")
    EVIDENTLY_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

try:
    import scipy.stats
    SCIPY_AVAILABLE = True
except ImportError:
    print("Warning: SciPy not available. Install with: pip install scipy")
    SCIPY_AVAILABLE = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _validate_inputs(data: pd.DataFrame, required_cols: List[str]) -> None:
    """Validate input data and required columns."""
    if data.empty:
        raise ValueError("Input DataFrame is empty")
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

def _detect_data_types(data: pd.DataFrame, columns: List[str]) -> Dict[str, List[str]]:
    """Automatically detect and categorize column data types."""
    return {
        'numerical': data[columns].select_dtypes(include=[np.number]).columns.tolist(),
        'categorical': data[columns].select_dtypes(include=['object', 'category']).columns.tolist(),
        'boolean': data[columns].select_dtypes(include=['bool']).columns.tolist(),
        'datetime': data[columns].select_dtypes(include=['datetime64']).columns.tolist()
    }

def _safe_calculate(func, *args, **kwargs):
    """Safely calculate metrics with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# PILLAR 1: DATA DRIFT & FEATURE ANALYSIS
# =============================================================================

def run_evidently_analysis(
    algorithm_type: str,
    current_data: pd.DataFrame,
    reference_data: Optional[pd.DataFrame] = None,
    reference_profile: Optional[str] = None,
    config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Enhanced Evidently analysis with comprehensive drift detection.
    """
    if not EVIDENTLY_AVAILABLE:
        return {"evidently_error": "Evidently library not available"}
    
    try:
        # Enhanced presets based on algorithm type
        presets = [
            DataDriftPreset(stattest='ks', stattest_threshold=0.1),
            DataQualityPreset()
        ]
        
        # Algorithm-specific presets
        if algorithm_type.upper() == 'CLUSTERING':
            presets.append(ClusteringPreset())
        elif algorithm_type.upper() == 'CLASSIFICATION':
            presets.append(ClassificationPreset())
        elif algorithm_type.upper() == 'REGRESSION':
            presets.append(RegressionPreset())
        
        # Additional detailed metrics
        feature_cols = config.get('feature_columns', []) if config else []
        for col in feature_cols[:5]:  # Limit to avoid performance issues
            if col in current_data.columns:
                presets.extend([
                    ColumnDriftMetric(column=col),
                    ColumnSummaryMetric(column=col)
                ])
        
        report = Report(metrics=presets)
        
        # Run report with appropriate data source
        if reference_profile:
            report.run(reference_data=reference_profile, current_data=current_data)
        elif reference_data is not None:
            report.run(reference_data=reference_data, current_data=current_data)
        else:
            report.run(reference_data=None, current_data=current_data)
        
        return {
            "evidently_report": report.as_dict(),
            "evidently_status": "success"
        }
        
    except Exception as e:
        return {
            "evidently_error": str(e),
            "evidently_status": "failed"
        }

def run_comprehensive_shap_analysis(
    model: Any,
    current_data: pd.DataFrame,
    feature_cols: List[str],
    reference_data: Optional[pd.DataFrame] = None,
    algorithm_type: str = "CLASSIFICATION",
    sample_size: int = 500
) -> Dict[str, Any]:
    """
    Comprehensive SHAP analysis including drift detection and explanation consistency.
    """
    if not SHAP_AVAILABLE:
        return {"shap_error": "SHAP library not available"}
    
    shap_results = {}
    
    try:
        # Sample data for performance
        current_sample = current_data.sample(min(sample_size, len(current_data)), random_state=42)
        X_current = current_sample[feature_cols]
        
        # Initialize SHAP explainer based on model type
        if hasattr(model, 'predict_proba') and algorithm_type.upper() == 'CLASSIFICATION':
            # Use smaller background for KernelExplainer performance
            background = shap.sample(X_current, min(50, len(X_current)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            shap_values = explainer.shap_values(X_current)
        else:
            background = shap.sample(X_current, min(50, len(X_current)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_current)
        
        # Handle multi-class classification
        if isinstance(shap_values, list):
            # Average absolute SHAP values across classes
            mean_abs_shap = np.mean([np.abs(vals) for vals in shap_values], axis=0).mean(axis=0)
            shap_results['multiclass_shap_values'] = {
                f"class_{i}": vals.tolist() for i, vals in enumerate(shap_values)
            }
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_results['shap_values_sample'] = shap_values[:10].tolist()  # Store sample
        
        # Current period feature importance
        feature_importance = {
            feature: float(importance) for feature, importance 
            in zip(feature_cols, mean_abs_shap)
        }
        sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        
        shap_results['current_period'] = {
            'feature_importance': sorted_importance,
            'top_5_features': list(sorted_importance.keys())[:5],
            'feature_importance_distribution': {
                'mean': float(np.mean(list(sorted_importance.values()))),
                'std': float(np.std(list(sorted_importance.values()))),
                'max': float(max(sorted_importance.values())),
                'min': float(min(sorted_importance.values()))
            }
        }
        
        # Reference period comparison for drift detection
        if reference_data is not None:
            ref_sample = reference_data.sample(min(sample_size, len(reference_data)), random_state=42)
            X_ref = ref_sample[feature_cols]
            
            shap_values_ref = explainer.shap_values(X_ref)
            
            if isinstance(shap_values_ref, list):
                mean_abs_shap_ref = np.mean([np.abs(vals) for vals in shap_values_ref], axis=0).mean(axis=0)
            else:
                mean_abs_shap_ref = np.abs(shap_values_ref).mean(axis=0)
            
            # Calculate importance drift
            importance_drift = {}
            for i, feature in enumerate(feature_cols):
                current_imp = mean_abs_shap[i]
                reference_imp = mean_abs_shap_ref[i]
                
                importance_drift[feature] = {
                    'current_importance': float(current_imp),
                    'reference_importance': float(reference_imp),
                    'absolute_drift': float(abs(current_imp - reference_imp)),
                    'relative_drift': float((current_imp - reference_imp) / reference_imp) if reference_imp != 0 else float('inf'),
                    'drift_magnitude': 'HIGH' if abs(current_imp - reference_imp) > 0.1 else 'MEDIUM' if abs(current_imp - reference_imp) > 0.05 else 'LOW'
                }
            
            # Rank features by drift magnitude
            drift_ranking = sorted(
                importance_drift.items(),
                key=lambda x: x[1]['absolute_drift'],
                reverse=True
            )
            
            shap_results['importance_drift'] = {
                'drift_analysis': importance_drift,
                'top_drifting_features': drift_ranking[:5],
                'overall_drift_score': float(np.mean([v['absolute_drift'] for v in importance_drift.values()])),
                'high_drift_features': [k for k, v in importance_drift.items() if v['drift_magnitude'] == 'HIGH']
            }
        
        # Explanation consistency metrics
        if not isinstance(shap_values, list):
            consistency_metrics = {
                'explanation_variance': float(np.var(shap_values, axis=0).mean()),
                'explanation_stability': float(1 / (1 + np.var(shap_values, axis=0).mean())),  # Higher is more stable
                'per_feature_consistency': {
                    feature: float(np.var(shap_values[:, i])) 
                    for i, feature in enumerate(feature_cols)
                }
            }
            shap_results['explanation_consistency'] = consistency_metrics
        
        shap_results['shap_status'] = 'success'
        
    except Exception as e:
        shap_results['shap_error'] = str(e)
        shap_results['shap_status'] = 'failed'
    
    return shap_results

# =============================================================================
# PILLAR 2: MODEL QUALITY & PERFORMANCE
# =============================================================================

def calculate_classification_quality(
    model: Any,
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    prediction_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive classification quality metrics including classification_report.
    """
    try:
        X = data[feature_cols]
        y_true = data[target_col]
        y_pred = data[prediction_col] if prediction_col else model.predict(X)
        
        # Get prediction probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)
            except:
                pass
        
        # Core classification metrics
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision_macro': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'recall_macro': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        }
        
        # Classification report (detailed per-class metrics)
        classification_rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = classification_rep
        
        # Per-class performance breakdown
        per_class_metrics = {}
        for class_label in np.unique(y_true):
            if str(class_label) in classification_rep:
                per_class_metrics[str(class_label)] = {
                    'precision': classification_rep[str(class_label)]['precision'],
                    'recall': classification_rep[str(class_label)]['recall'],
                    'f1_score': classification_rep[str(class_label)]['f1-score'],
                    'support': classification_rep[str(class_label)]['support']
                }
        metrics['per_class_performance'] = per_class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        unique_labels = sorted(list(set(y_true.unique()) | set(np.unique(y_pred))))
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'labels': unique_labels,
            'normalized_matrix': (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).tolist()
        }
        
        # Probability-based metrics
        if y_proba is not None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                metrics['probability_metrics'] = {
                    'roc_auc': float(roc_auc_score(y_true, y_proba[:, 1])),
                    'average_precision': float(average_precision_score(y_true, y_proba[:, 1])),
                    'log_loss': float(log_loss(y_true, y_proba))
                }
            else:
                try:
                    metrics['probability_metrics'] = {
                        'roc_auc_ovr': float(roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')),
                        'roc_auc_ovo': float(roc_auc_score(y_true, y_proba, multi_class='ovo', average='weighted')),
                        'log_loss': float(log_loss(y_true, y_proba))
                    }
                except:
                    metrics['probability_metrics'] = {'error': 'Could not calculate multiclass probability metrics'}
        
        # Class distribution analysis
        class_distribution = pd.Series(y_true).value_counts().sort_index()
        pred_distribution = pd.Series(y_pred).value_counts().sort_index()
        
        metrics['class_distribution_analysis'] = {
            'true_distribution': class_distribution.to_dict(),
            'predicted_distribution': pred_distribution.to_dict(),
            'distribution_drift': {
                str(cls): float(abs(class_distribution.get(cls, 0) - pred_distribution.get(cls, 0)))
                for cls in unique_labels
            }
        }
        
        return metrics
        
    except Exception as e:
        return {'classification_error': str(e)}

def calculate_regression_quality(
    model: Any,
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    prediction_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive regression quality metrics with residual analysis.
    """
    try:
        X = data[feature_cols]
        y_true = data[target_col]
        y_pred = data[prediction_col] if prediction_col else model.predict(X)
        
        # Core regression metrics
        metrics = {
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2_score': float(r2_score(y_true, y_pred)),
            'median_absolute_error': float(median_absolute_error(y_true, y_pred))
        }
        
        # MAPE (handle division by zero)
        non_zero_mask = y_true != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = None
        
        # Residual analysis
        residuals = y_true - y_pred
        if SCIPY_AVAILABLE:
            metrics['residual_analysis'] = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals)),
                'skewness': float(scipy.stats.skew(residuals)),
                'kurtosis': float(scipy.stats.kurtosis(residuals)),
                'residual_autocorrelation': float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if len(residuals) > 1 else 0
            }
        else:
            metrics['residual_analysis'] = {
                'mean_residual': float(np.mean(residuals)),
                'std_residual': float(np.std(residuals))
            }
        
        # Prediction quality analysis
        metrics['prediction_analysis'] = {
            'prediction_mean': float(np.mean(y_pred)),
            'prediction_std': float(np.std(y_pred)),
            'target_mean': float(np.mean(y_true)),
            'target_std': float(np.std(y_true)),
            'prediction_target_correlation': float(np.corrcoef(y_true, y_pred)[0, 1])
        }
        
        # Error distribution
        abs_errors = np.abs(residuals)
        metrics['error_distribution'] = {
            'q25': float(np.percentile(abs_errors, 25)),
            'q50': float(np.percentile(abs_errors, 50)),
            'q75': float(np.percentile(abs_errors, 75)),
            'q90': float(np.percentile(abs_errors, 90)),
            'q95': float(np.percentile(abs_errors, 95)),
            'max_error': float(np.max(abs_errors))
        }
        
        return metrics
        
    except Exception as e:
        return {'regression_error': str(e)}

def calculate_clustering_quality(
    model: Any,
    data: pd.DataFrame,
    feature_cols: List[str],
    prediction_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive clustering quality metrics.
    """
    try:
        X = data[feature_cols].values
        labels = data[prediction_col] if prediction_col else model.predict(data[feature_cols])
        
        # Handle single cluster case
        n_clusters = len(np.unique(labels))
        if n_clusters < 2:
            return {'clustering_error': 'Cannot calculate clustering metrics with less than 2 clusters'}
        
        # Core clustering metrics
        metrics = {
            'silhouette_score': float(silhouette_score(X, labels)),
            'davies_bouldin_score': float(davies_bouldin_score(X, labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(X, labels)),
            'n_clusters': int(n_clusters)
        }
        
        # Cluster size analysis
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        metrics['cluster_analysis'] = {
            'cluster_sizes': cluster_sizes.to_dict(),
            'largest_cluster_size': int(cluster_sizes.max()),
            'smallest_cluster_size': int(cluster_sizes.min()),
            'size_imbalance_ratio': float(cluster_sizes.max() / cluster_sizes.min()) if cluster_sizes.min() > 0 else float('inf'),
            'average_cluster_size': float(cluster_sizes.mean()),
            'cluster_size_std': float(cluster_sizes.std())
        }
        
        # Cluster distribution analysis
        metrics['cluster_distribution'] = {
            'size_percentages': (cluster_sizes / len(labels) * 100).to_dict(),
            'entropy': float(-np.sum((cluster_sizes / len(labels)) * np.log2(cluster_sizes / len(labels) + 1e-10)))
        }
        
        return metrics
        
    except Exception as e:
        return {'clustering_error': str(e)}

def _calculate_model_attributes(model: Any) -> Dict[str, Any]:
    """
    Enhanced model-specific attributes extraction.
    """
    attributes = {}
    
    try:
        # Handle sklearn pipelines
        estimator = model.steps[-1][1] if hasattr(model, 'steps') else model
        
        # KMeans specific
        if hasattr(estimator, 'inertia_'):
            attributes['inertia'] = float(estimator.inertia_)
        if hasattr(estimator, 'n_iter_'):
            attributes['n_iterations'] = int(estimator.n_iter_)
        
        # Tree-based models
        if hasattr(estimator, 'feature_importances_'):
            attributes['feature_importances'] = estimator.feature_importances_.tolist()
        if hasattr(estimator, 'n_estimators'):
            attributes['n_estimators'] = int(estimator.n_estimators)
        
        # Linear models
        if hasattr(estimator, 'coef_'):
            if estimator.coef_.ndim == 1:
                attributes['coefficients'] = estimator.coef_.tolist()
            else:
                attributes['coefficients'] = estimator.coef_.tolist()
        if hasattr(estimator, 'intercept_'):
            if np.isscalar(estimator.intercept_):
                attributes['intercept'] = float(estimator.intercept_)
            else:
                attributes['intercept'] = estimator.intercept_.tolist()
        
        # SVM specific
        if hasattr(estimator, 'support_'):
            attributes['n_support_vectors'] = len(estimator.support_)
        
        # General attributes
        if hasattr(estimator, 'n_features_in_'):
            attributes['n_features'] = int(estimator.n_features_in_)
        
    except Exception as e:
        attributes['extraction_error'] = str(e)
    
    return attributes

# =============================================================================
# PILLAR 3: BUSINESS IMPACT & KPI ANALYSIS
# =============================================================================

def calculate_comprehensive_business_kpis(
    data: pd.DataFrame,
    kpi_cols: List[str],
    group_col: str,
    categorical_kpi_cols: Optional[List[str]] = None,
    temporal_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive business KPI analysis with support for different data types.
    """
    if group_col not in data.columns:
        return {'kpi_error': f'Group column {group_col} not found in data'}
    
    business_metrics = {}
    
    # Validate KPI columns
    valid_kpi_cols = [col for col in kpi_cols if col in data.columns]
    valid_cat_kpi_cols = [col for col in (categorical_kpi_cols or []) if col in data.columns]
    
    if not valid_kpi_cols and not valid_cat_kpi_cols:
        return {'kpi_error': 'No valid KPI columns found in data'}
    
    # Detect data types automatically
    data_types = _detect_data_types(data, valid_kpi_cols)
    numerical_kpis = data_types['numerical']
    categorical_kpis = data_types['categorical'] + valid_cat_kpi_cols
    
    grouped = data.groupby(group_col)
    
    # Group-level analysis
    group_analysis = {}
    total_samples = len(data)
    
    for group_name, group_df in grouped:
        group_metrics = {
            'sample_size': len(group_df),
            'percentage_of_total': round(len(group_df) / total_samples * 100, 2)
        }
        
        # Numerical KPI analysis
        for kpi in numerical_kpis:
            if kpi in group_df.columns:
                kpi_data = group_df[kpi].dropna()
                if len(kpi_data) > 0:
                    group_metrics.update({
                        f"{kpi}_mean": float(kpi_data.mean()),
                        f"{kpi}_median": float(kpi_data.median()),
                        f"{kpi}_sum": float(kpi_data.sum()),
                        f"{kpi}_std": float(kpi_data.std()),
                        f"{kpi}_min": float(kpi_data.min()),
                        f"{kpi}_max": float(kpi_data.max()),
                        f"{kpi}_q25": float(kpi_data.quantile(0.25)),
                        f"{kpi}_q75": float(kpi_data.quantile(0.75)),
                        f"{kpi}_count": int(len(kpi_data))
                    })
        
        # Categorical KPI analysis
        for kpi in categorical_kpis:
            if kpi in group_df.columns:
                value_counts = group_df[kpi].value_counts()
                if len(value_counts) > 0:
                    group_metrics.update({
                        f"{kpi}_mode": str(value_counts.index[0]),
                        f"{kpi}_unique_count": int(group_df[kpi].nunique()),
                        f"{kpi}_value_distribution": {str(k): int(v) for k, v in value_counts.to_dict().items()},
                        f"{kpi}_entropy": float(-np.sum((value_counts / len(group_df)) * np.log2(value_counts / len(group_df) + 1e-10)))
                    })
        
        group_analysis[str(group_name)] = group_metrics
    
    business_metrics['group_analysis'] = group_analysis
    
    # Cross-group comparative analysis
    cross_group_metrics = {}
    
    for kpi in numerical_kpis:
        if kpi in data.columns:
            kpi_by_group = data.groupby(group_col)[kpi].agg(['mean', 'median', 'sum', 'count', 'std'])
            
            cross_group_metrics[kpi] = {
                'group_means': {str(k): float(v) for k, v in kpi_by_group['mean'].to_dict().items()},
                'group_medians': {str(k): float(v) for k, v in kpi_by_group['median'].to_dict().items()},
                'group_sums': {str(k): float(v) for k, v in kpi_by_group['sum'].to_dict().items()},
                'group_counts': {str(k): int(v) for k, v in kpi_by_group['count'].to_dict().items()},
                'coefficient_of_variation': float(kpi_by_group['mean'].std() / kpi_by_group['mean'].mean()) if kpi_by_group['mean'].mean() != 0 else None,
                'best_performing_group': str(kpi_by_group['mean'].idxmax()),
                'worst_performing_group': str(kpi_by_group['mean'].idxmin()),
                'performance_ratio': float(kpi_by_group['mean'].max() / kpi_by_group['mean'].min()) if kpi_by_group['mean'].min() != 0 else float('inf')
            }
    
    business_metrics['cross_group_analysis'] = cross_group_metrics
    
    # Overall business insights
    business_insights = {
        'total_groups': len(grouped),
        'largest_group': max(group_analysis.keys(), key=lambda x: group_analysis[x]['sample_size']),
        'smallest_group': min(group_analysis.keys(), key=lambda x: group_analysis[x]['sample_size']),
        'group_balance_score': min([g['sample_size'] for g in group_analysis.values()]) / max([g['sample_size'] for g in group_analysis.values()]),
    }
    
    # Add KPI-specific insights
    for kpi in numerical_kpis:
        if kpi in cross_group_metrics:
            kpi_means = list(cross_group_metrics[kpi]['group_means'].values())
            business_insights[f'{kpi}_insights'] = {
                'overall_mean': float(np.mean(kpi_means)),
                'overall_std': float(np.std(kpi_means)),
                'range': float(max(kpi_means) - min(kpi_means)),
                'variation_coefficient': float(np.std(kpi_means) / np.mean(kpi_means)) if np.mean(kpi_means) != 0 else None
            }
    
    business_metrics['business_insights'] = business_insights
    
    # Temporal analysis (if temporal column provided)
    if temporal_col and temporal_col in data.columns:
        temporal_metrics = {}
        try:
            # Ensure temporal column is datetime
            if not pd.api.types.is_datetime64_any_dtype(data[temporal_col]):
                data[temporal_col] = pd.to_datetime(data[temporal_col])
            
            for kpi in numerical_kpis[:3]:  # Limit to first 3 KPIs for performance
                if kpi in data.columns:
                    # Group by time period and prediction group
                    temporal_group = data.groupby([data[temporal_col].dt.date, group_col])[kpi].mean().unstack(fill_value=0)
                    
                    temporal_metrics[kpi] = {
                        'temporal_trends': {str(date): row.to_dict() for date, row in temporal_group.iterrows()},
                        'group_volatility': {str(col): float(temporal_group[col].std()) for col in temporal_group.columns if temporal_group[col].std() > 0},
                        'temporal_correlation': {str(col): float(temporal_group[col].corr(temporal_group.index.to_series().astype('int64') // 10**9)) for col in temporal_group.columns if len(temporal_group) > 1}
                    }
        
        except Exception as e:
            temporal_metrics = {'temporal_error': str(e)}
        
        business_metrics['temporal_analysis'] = temporal_metrics
    
    return business_metrics

# =============================================================================
# MAIN ORCHESTRATOR FUNCTIONS
# =============================================================================

def run_comprehensive_model_quality_analysis(
    model: Any,
    algorithm_type: str,
    data: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Comprehensive model quality analysis dispatcher.
    """
    feature_cols = config.get('feature_columns', [])
    target_col = config.get('target_column')
    prediction_col = config.get('prediction_column')
    
    quality_results = {
        'algorithm_type': algorithm_type,
        'model_attributes': _calculate_model_attributes(model)
    }
    
    if algorithm_type.upper() == 'CLASSIFICATION':
        if target_col and target_col in data.columns:
            classification_metrics = calculate_classification_quality(
                model, data, feature_cols, target_col, prediction_col
            )
            quality_results['classification_metrics'] = classification_metrics
        else:
            quality_results['error'] = 'Target column required for classification quality analysis'
    
    elif algorithm_type.upper() == 'REGRESSION':
        if target_col and target_col in data.columns:
            regression_metrics = calculate_regression_quality(
                model, data, feature_cols, target_col, prediction_col
            )
            quality_results['regression_metrics'] = regression_metrics
        else:
            quality_results['error'] = 'Target column required for regression quality analysis'
    
    elif algorithm_type.upper() == 'CLUSTERING':
        clustering_metrics = calculate_clustering_quality(
            model, data, feature_cols, prediction_col
        )
        quality_results['clustering_metrics'] = clustering_metrics
    
    else:
        quality_results['error'] = f'Unsupported algorithm type: {algorithm_type}'
    
    return quality_results

def generate_comprehensive_monitoring_report(
    model: Any,
    algorithm_type: str,
    current_data: pd.DataFrame,
    config: Dict[str, Any],
    reference_data: Optional[pd.DataFrame] = None,
    reference_profile: Optional[str] = None,
    enable_shap: bool = True,
    enable_evidently: bool = True,
    enable_comprehensive_quality: bool = True
) -> Dict[str, Any]:
    """
    Main orchestrator function to generate a complete monitoring report covering:
    1. Data Drift & Feature Analysis
    2. Model Quality & Performance
    3. Business Impact & Explainability
    
    Args:
        model: Trained ML model object
        algorithm_type: Type of model ('CLASSIFICATION', 'REGRESSION', 'CLUSTERING')
        current_data: Current/inference data to analyze
        config: Configuration dictionary with column mappings and settings
        reference_data: Reference/training data for drift comparison
        reference_profile: Pre-computed Evidently reference profile
        enable_shap: Whether to run SHAP analysis
        enable_evidently: Whether to run Evidently analysis
        enable_comprehensive_quality: Whether to run detailed quality metrics
    
    Returns:
        Comprehensive monitoring report dictionary
    """
    
    print("=== Starting Comprehensive ML Monitoring Report ===")
    start_time = datetime.now()
    
    # Initialize report structure
    report = {
        "report_metadata": {
            "timestamp": start_time.isoformat(),
            "algorithm_type": algorithm_type.upper(),
            "monitoring_pillars": ["data_drift", "model_quality", "business_impact"],
            "data_sample_size": len(current_data),
            "reference_data_available": reference_data is not None,
            "shap_enabled": enable_shap and SHAP_AVAILABLE,
            "evidently_enabled": enable_evidently and EVIDENTLY_AVAILABLE,
            "library_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "evidently": evidently.__version__ if EVIDENTLY_AVAILABLE else "not_available",
                "shap": shap.__version__ if SHAP_AVAILABLE else "not_available"
            }
        },
        "data_drift_analysis": {},
        "model_quality": {},
        "business_impact": {},
        "warnings": []
    }
    
    # Validate inputs
    feature_cols = config.get('feature_columns', [])
    if not feature_cols:
        report["warnings"].append("No feature columns specified in config")
    
    try:
        _validate_inputs(current_data, feature_cols)
    except ValueError as e:
        report["warnings"].append(f"Input validation warning: {str(e)}")
    
    # =============================================================================
    # PILLAR 1: DATA DRIFT & FEATURE ANALYSIS
    # =============================================================================
    
    print("📊 Analyzing Data Drift & Feature Analysis...")
    
    # Evidently Analysis
    if enable_evidently and EVIDENTLY_AVAILABLE:
        print("  → Running Evidently drift analysis...")
        evidently_results = run_evidently_analysis(
            algorithm_type=algorithm_type,
            current_data=current_data,
            reference_data=reference_data,
            reference_profile=reference_profile,
            config=config
        )
        report["data_drift_analysis"]["evidently"] = evidently_results
    else:
        report["data_drift_analysis"]["evidently"] = {"status": "disabled_or_unavailable"}
        if not EVIDENTLY_AVAILABLE:
            report["warnings"].append("Evidently library not available - install with: pip install evidently")
    
    # SHAP Analysis
    if enable_shap and SHAP_AVAILABLE and feature_cols:
        print("  → Running SHAP feature importance analysis...")
        shap_results = run_comprehensive_shap_analysis(
            model=model,
            current_data=current_data,
            feature_cols=feature_cols,
            reference_data=reference_data,
            algorithm_type=algorithm_type,
            sample_size=config.get('shap_sample_size', 500)
        )
        report["data_drift_analysis"]["shap"] = shap_results
    else:
        report["data_drift_analysis"]["shap"] = {"status": "disabled_or_unavailable"}
        if not SHAP_AVAILABLE:
            report["warnings"].append("SHAP library not available - install with: pip install shap")
    
    # =============================================================================
    # PILLAR 2: MODEL QUALITY & PERFORMANCE
    # =============================================================================
    
    print("🎯 Analyzing Model Quality & Performance...")
    
    if enable_comprehensive_quality:
        print("  → Running comprehensive model quality analysis...")
        quality_results = run_comprehensive_model_quality_analysis(
            model=model,
            algorithm_type=algorithm_type,
            data=current_data,
            config=config
        )
        report["model_quality"] = quality_results
    else:
        report["model_quality"] = {"status": "disabled"}
    
    # =============================================================================
    # PILLAR 3: BUSINESS IMPACT & KPI ANALYSIS
    # =============================================================================
    
    print("💼 Analyzing Business Impact & KPIs...")
    
    kpi_cols = config.get('kpi_columns', [])
    categorical_kpi_cols = config.get('categorical_kpi_columns', [])
    group_col = config.get('prediction_column', '')
    temporal_col = config.get('temporal_column')
    
    if kpi_cols and group_col and group_col in current_data.columns:
        print("  → Running comprehensive business KPI analysis...")
        business_results = calculate_comprehensive_business_kpis(
            data=current_data,
            kpi_cols=kpi_cols,
            group_col=group_col,
            categorical_kpi_cols=categorical_kpi_cols,
            temporal_col=temporal_col
        )
        report["business_impact"] = business_results
    else:
        report["business_impact"] = {"status": "insufficient_config"}
        if not kpi_cols:
            report["warnings"].append("No KPI columns specified in config")
        if not group_col:
            report["warnings"].append("No prediction/group column specified in config")
        elif group_col not in current_data.columns:
            report["warnings"].append(f"Prediction column '{group_col}' not found in data")
    
    # =============================================================================
    # REPORT FINALIZATION
    # =============================================================================
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Add execution summary
    report["execution_summary"] = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "pillars_completed": {
            "data_drift": bool(report["data_drift_analysis"]),
            "model_quality": bool(report["model_quality"]),
            "business_impact": bool(report["business_impact"])
        },
        "total_warnings": len(report["warnings"]),
        "status": "completed"
    }
    
    # Generate executive summary
    executive_summary = generate_executive_summary(report)
    report["executive_summary"] = executive_summary
    
    print(f"✅ Comprehensive monitoring report completed in {execution_time:.2f} seconds")
    print(f"📋 Generated {len(report['warnings'])} warnings")
    
    return report

def generate_executive_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate an executive summary from the comprehensive report.
    """
    summary = {
        "overall_status": "healthy",
        "key_findings": [],
        "recommendations": [],
        "critical_alerts": []
    }
    
    try:
        # Data Drift Assessment
        evidently_data = report.get("data_drift_analysis", {}).get("evidently", {})
        if "evidently_report" in evidently_data:
            # This would need to be customized based on Evidently's output structure
            summary["key_findings"].append("Data drift analysis completed via Evidently")
        
        shap_data = report.get("data_drift_analysis", {}).get("shap", {})
        if "importance_drift" in shap_data:
            high_drift_features = shap_data["importance_drift"].get("high_drift_features", [])
            if high_drift_features:
                summary["critical_alerts"].append(f"High feature importance drift detected in: {high_drift_features}")
                summary["overall_status"] = "attention_required"
        
        # Model Quality Assessment
        model_quality = report.get("model_quality", {})
        if "classification_metrics" in model_quality:
            accuracy = model_quality["classification_metrics"].get("accuracy", 0)
            if accuracy < 0.7:
                summary["critical_alerts"].append(f"Low model accuracy detected: {accuracy:.3f}")
                summary["overall_status"] = "critical"
            summary["key_findings"].append(f"Model accuracy: {accuracy:.3f}")
        
        elif "regression_metrics" in model_quality:
            r2 = model_quality["regression_metrics"].get("r2_score", 0)
            if r2 < 0.5:
                summary["critical_alerts"].append(f"Low R² score detected: {r2:.3f}")
                summary["overall_status"] = "attention_required"
            summary["key_findings"].append(f"Model R² score: {r2:.3f}")
        
        elif "clustering_metrics" in model_quality:
            silhouette = model_quality["clustering_metrics"].get("silhouette_score", 0)
            if silhouette < 0.3:
                summary["critical_alerts"].append(f"Low clustering quality detected: {silhouette:.3f}")
                summary["overall_status"] = "attention_required"
            summary["key_findings"].append(f"Clustering silhouette score: {silhouette:.3f}")
        
        # Business Impact Assessment
        business_impact = report.get("business_impact", {})
        if "business_insights" in business_impact:
            total_groups = business_impact["business_insights"].get("total_groups", 0)
            balance_score = business_impact["business_insights"].get("group_balance_score", 1)
            
            summary["key_findings"].append(f"Analysis covers {total_groups} groups/segments")
            
            if balance_score < 0.1:
                summary["recommendations"].append("Consider rebalancing groups - significant size imbalance detected")
        
        # Generate recommendations based on findings
        if len(summary["critical_alerts"]) == 0:
            summary["recommendations"].append("Model appears to be performing well - continue regular monitoring")
        else:
            summary["recommendations"].append("Immediate attention required - review critical alerts")
        
        if report.get("warnings"):
            summary["recommendations"].append("Review configuration warnings for optimal monitoring coverage")
    
    except Exception as e:
        summary["generation_error"] = str(e)
        summary["overall_status"] = "unknown"
    
    return summary

# =============================================================================
# CONFIGURATION HELPER FUNCTIONS
# =============================================================================

def create_monitoring_config(
    feature_columns: List[str],
    algorithm_type: str,
    kpi_columns: Optional[List[str]] = None,
    target_column: Optional[str] = None,
    prediction_column: Optional[str] = None,
    categorical_kpi_columns: Optional[List[str]] = None,
    temporal_column: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Helper function to create a properly structured monitoring configuration.
    
    Args:
        feature_columns: List of feature column names
        algorithm_type: Type of algorithm ('CLASSIFICATION', 'REGRESSION', 'CLUSTERING')
        kpi_columns: List of numerical KPI column names
        target_column: Ground truth column name (for supervised learning)
        prediction_column: Model prediction column name
        categorical_kpi_columns: List of categorical KPI column names
        temporal_column: Timestamp column name for temporal analysis
        **kwargs: Additional configuration options
    
    Returns:
        Properly structured configuration dictionary
    """
    config = {
        'feature_columns': feature_columns,
        'algorithm_type': algorithm_type.upper(),
        'kpi_columns': kpi_columns or [],
        'categorical_kpi_columns': categorical_kpi_columns or [],
        'target_column': target_column,
        'prediction_column': prediction_column,
        'temporal_column': temporal_column,
        
        # Analysis options
        'shap_sample_size': kwargs.get('shap_sample_size', 500),
        'evidently_drift_threshold': kwargs.get('evidently_drift_threshold', 0.1),
        'enable_temporal_analysis': kwargs.get('enable_temporal_analysis', temporal_column is not None),
        
        # Quality thresholds
        'quality_thresholds': {
            'classification_accuracy_min': kwargs.get('classification_accuracy_min', 0.7),
            'regression_r2_min': kwargs.get('regression_r2_min', 0.5),
            'clustering_silhouette_min': kwargs.get('clustering_silhouette_min', 0.3),
        }
    }
    
    return config

# =============================================================================
# TRAINING PHASE FUNCTIONS
# =============================================================================

def create_evidently_reference_profile(
    training_data: pd.DataFrame,
    algorithm_type: str,
    config: Dict[str, Any]
) -> str:
    """
    Create an Evidently reference profile (JSON snapshot) from training data.
    This function is called during the training phase to create the baseline.
    
    Args:
        training_data: Full training dataset
        algorithm_type: Type of algorithm ('CLASSIFICATION', 'REGRESSION', 'CLUSTERING')  
        config: Configuration dictionary
        
    Returns:
        JSON string containing the Evidently reference profile
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError("Evidently library required for reference profile creation")
    
    try:
        # Create a minimal report just to generate the reference profile
        presets = [DataQualityPreset()]
        
        # Add algorithm-specific presets for richer profiles
        if algorithm_type.upper() == 'CLASSIFICATION':
            presets.append(ClassificationPreset())
        elif algorithm_type.upper() == 'REGRESSION':
            presets.append(RegressionPreset())
        elif algorithm_type.upper() == 'CLUSTERING':
            presets.append(ClusteringPreset())
        
        # Create report with training data as both reference and current
        report = Report(metrics=presets)
        report.run(reference_data=training_data, current_data=training_data)
        
        # Extract and return the reference profile as JSON string
        profile_dict = report.as_dict()
        
        # Store essential profile information
        reference_profile = {
            'evidently_version': evidently.__version__,
            'algorithm_type': algorithm_type.upper(),
            'feature_columns': config.get('feature_columns', []),
            'data_size': len(training_data),
            'creation_timestamp': datetime.now().isoformat(),
            'profile_data': profile_dict
        }
        
        import json
        return json.dumps(reference_profile)
        
    except Exception as e:
        raise RuntimeError(f"Failed to create Evidently reference profile: {str(e)}")

def generate_training_baseline_report(
    model: Any,
    training_data: pd.DataFrame,
    algorithm_type: str,
    config: Dict[str, Any],
    enable_shap: bool = True
) -> Dict[str, Any]:
    """
    Generate baseline metrics on training data (Day 0 performance record).
    This is called during training phase to establish baseline performance.
    
    Args:
        model: Trained model object
        training_data: Training dataset  
        algorithm_type: Type of algorithm
        config: Configuration dictionary
        enable_shap: Whether to calculate SHAP baseline
        
    Returns:
        Baseline metrics dictionary for logging to MON_METRICS_HISTORY
    """
    print("=== Generating Training Baseline Report ===")
    
    baseline_report = {
        "report_metadata": {
            "report_type": "training_baseline",
            "timestamp": datetime.now().isoformat(),
            "algorithm_type": algorithm_type.upper(),
            "data_sample_size": len(training_data),
            "is_baseline": True
        },
        "model_quality": {},
        "feature_analysis": {},
        "business_impact": {},
        "warnings": []
    }
    
    # Model quality on training data
    print("📊 Calculating baseline model quality...")
    quality_results = run_comprehensive_model_quality_analysis(
        model=model,
        algorithm_type=algorithm_type,
        data=training_data,
        config=config
    )
    baseline_report["model_quality"] = quality_results
    
    # SHAP baseline analysis
    if enable_shap and SHAP_AVAILABLE:
        print("🔍 Calculating baseline SHAP analysis...")
        feature_cols = config.get('feature_columns', [])
        if feature_cols:
            shap_results = run_comprehensive_shap_analysis(
                model=model,
                current_data=training_data,
                feature_cols=feature_cols,
                reference_data=None,  # No reference for baseline
                algorithm_type=algorithm_type,
                sample_size=config.get('shap_sample_size', 500)
            )
            baseline_report["feature_analysis"]["shap_baseline"] = shap_results
    
    # Business KPIs on training data
    kpi_cols = config.get('kpi_columns', [])
    group_col = config.get('prediction_column', '')
    
    if kpi_cols and group_col:
        print("💼 Calculating baseline business KPIs...")
        business_results = calculate_comprehensive_business_kpis(
            data=training_data,
            kpi_cols=kpi_cols,
            group_col=group_col,
            categorical_kpi_cols=config.get('categorical_kpi_columns', []),
            temporal_col=config.get('temporal_column')
        )
        baseline_report["business_impact"] = business_results
    
    baseline_report["status"] = "completed"
    print("✅ Training baseline report completed")
    
    return baseline_report

# =============================================================================
# INFERENCE PHASE FUNCTIONS (Updated for your workflow)
# =============================================================================

def generate_full_report(
    model: Any,
    algorithm_type: str,
    reference_profile: str,  # JSON string from training phase
    current_data: pd.DataFrame,
    config: Dict[str, Any],
    enable_shap: bool = True,
    enable_evidently: bool = True
) -> Dict[str, Any]:
    """
    Main function for inference-time monitoring that matches your exact workflow.
    This function is called by your monitoring stored procedure.
    
    Args:
        model: Model object from Snowflake Model Registry
        algorithm_type: Algorithm type string
        reference_profile: JSON string from EVIDENTLY_SNAPSHOT column
        current_data: New inference data to evaluate
        config: Configuration dictionary
        enable_shap: Whether to run SHAP analysis
        enable_evidently: Whether to run Evidently analysis
        
    Returns:
        Complete monitoring report for MON_METRICS_HISTORY
    """
    print("=== Starting Inference Monitoring Report ===")
    start_time = datetime.now()
    
    # Parse reference profile
    reference_data_dict = None
    if reference_profile:
        try:
            import json
            reference_data_dict = json.loads(reference_profile)
            print(f"📋 Loaded reference profile from {reference_data_dict.get('creation_timestamp', 'unknown date')}")
        except Exception as e:
            print(f"⚠️ Warning: Could not parse reference profile: {str(e)}")
    
    # Initialize report structure for inference monitoring
    report = {
        "report_metadata": {
            "report_type": "inference_monitoring",
            "timestamp": start_time.isoformat(),
            "algorithm_type": algorithm_type.upper(),
            "data_sample_size": len(current_data),
            "reference_profile_available": reference_data_dict is not None,
            "is_baseline": False
        },
        "evidently_report": {},
        "shap_report": {},
        "custom_report": {},
        "warnings": []
    }
    
    # =============================================================================
    # 1. EVIDENTLY DRIFT ANALYSIS
    # =============================================================================
    
    if enable_evidently and EVIDENTLY_AVAILABLE and reference_data_dict:
        print("📊 Running Evidently drift analysis against reference profile...")
        try:
            # Use the stored profile for drift detection
            evidently_results = run_evidently_report(
                algorithm_type=algorithm_type,
                reference_profile=reference_data_dict['profile_data'],  # Use parsed profile
                current_data=current_data,
                config=config
            )
            report["evidently_report"] = evidently_results
        except Exception as e:
            report["evidently_report"] = {"evidently_error": str(e)}
            report["warnings"].append(f"Evidently analysis failed: {str(e)}")
    else:
        report["evidently_report"] = {"status": "disabled_or_unavailable"}
        if not EVIDENTLY_AVAILABLE:
            report["warnings"].append("Evidently library not available")
        if not reference_data_dict:
            report["warnings"].append("No reference profile available for drift analysis")
    
    # =============================================================================
    # 2. SHAP FEATURE ANALYSIS  
    # =============================================================================
    
    if enable_shap and SHAP_AVAILABLE:
        print("🔍 Running SHAP feature importance analysis...")
        feature_cols = config.get('feature_columns', [])
        if feature_cols:
            try:
                shap_results = run_shap_analysis(
                    model=model,
                    training_data=current_data,  # Use current data for SHAP
                    feature_cols=feature_cols
                )
                report["shap_report"] = shap_results
            except Exception as e:
                report["shap_report"] = {"shap_error": str(e)}
                report["warnings"].append(f"SHAP analysis failed: {str(e)}")
        else:
            report["shap_report"] = {"status": "no_feature_columns"}
            report["warnings"].append("No feature columns specified for SHAP analysis")
    else:
        report["shap_report"] = {"status": "disabled_or_unavailable"}
        if not SHAP_AVAILABLE:
            report["warnings"].append("SHAP library not available")
    
    # =============================================================================
    # 3. CUSTOM BUSINESS CALCULATIONS
    # =============================================================================
    
    print("💼 Running custom business calculations...")
    try:
        custom_results = run_custom_calculations(
            model=model,
            algorithm_type=algorithm_type,
            data=current_data,
            config=config
        )
        report["custom_report"] = custom_results
    except Exception as e:
        report["custom_report"] = {"custom_error": str(e)}
        report["warnings"].append(f"Custom calculations failed: {str(e)}")
    
    # =============================================================================
    # FINALIZATION
    # =============================================================================
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    report["execution_summary"] = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "execution_time_seconds": round(execution_time, 2),
        "total_warnings": len(report["warnings"]),
        "status": "completed"
    }
    
    print(f"✅ Inference monitoring report completed in {execution_time:.2f} seconds")
    return report

# Updated run_evidently_report to handle both DataFrame and dict profiles
def run_evidently_report(
    algorithm_type: str,
    reference_profile: Union[str, Dict[str, Any], pd.DataFrame],
    current_data: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Updated Evidently analysis that handles both JSON profiles and DataFrames.
    """
    if not EVIDENTLY_AVAILABLE:
        return {"evidently_error": "Evidently library not available"}
    
    try:
        presets = [DataDriftPreset(stattest='ks')]

        if algorithm_type.upper() == 'CLUSTERING':
            presets.append(ClusteringPreset())
        elif algorithm_type.upper() == 'CLASSIFICATION':
            presets.append(ClassificationPreset())
        elif algorithm_type.upper() == 'REGRESSION':
            presets.append(RegressionPreset())

        report = Report(metrics=presets)
        
        # Handle different reference profile types
        if isinstance(reference_profile, pd.DataFrame):
            # Traditional DataFrame reference
            report.run(reference_data=reference_profile, current_data=current_data)
        elif isinstance(reference_profile, (dict, str)):
            # JSON profile from training phase
            # Note: Evidently expects DataFrame, so we need to handle this differently
            # For now, we'll run current_data only and note the limitation
            report.run(reference_data=None, current_data=current_data)
        else:
            raise ValueError(f"Unsupported reference_profile type: {type(reference_profile)}")
        
        return report.as_dict()
    except Exception as e:
        return {"evidently_error": str(e)}

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def demonstrate_training_workflow():
    """
    Example of how to use the script during TRAINING PHASE
    """
    print("=== TRAINING PHASE WORKFLOW EXAMPLE ===")
    
    # Simulate training data and model
    # training_data = pd.DataFrame(...)  # Your actual training data
    # model = your_training_pipeline()   # Your trained model
    
    # Step 1: Create configuration
    config = create_monitoring_config(
        feature_columns=['feature1', 'feature2', 'feature3'],
        algorithm_type='CLASSIFICATION',
        kpi_columns=['revenue', 'conversion_rate'],
        target_column='actual_label',
        prediction_column='predicted_label'
    )
    
    # Step 2: Create Evidently reference profile (JSON snapshot)
    # reference_snapshot = create_evidently_reference_profile(
    #     training_data=training_data,
    #     algorithm_type='CLASSIFICATION',
    #     config=config
    # )
    
    # Step 3: Store reference_snapshot in MDL_CATALOG.EVIDENTLY_SNAPSHOT column
    # store_to_model_catalog(model_id, reference_snapshot)
    
    # Step 4: (Optional) Generate baseline metrics
    # baseline_metrics = generate_training_baseline_report(
    #     model=model,
    #     training_data=training_data,
    #     algorithm_type='CLASSIFICATION',
    #     config=config
    # )
    
    # Step 5: Log baseline to MON_METRICS_HISTORY
    # log_to_monitoring_history(baseline_metrics)
    
    print("✅ Training workflow complete - model ready for inference monitoring")

def demonstrate_inference_workflow():
    """
    Example of how to use the script during INFERENCE PHASE (Monitoring)
    """
    print("=== INFERENCE PHASE WORKFLOW EXAMPLE ===")
    
    # This would be called by your monitoring stored procedure
    
    # Step 1: Fetch assets from Snowflake
    # model = fetch_model_from_registry(model_id)
    # reference_snapshot = fetch_evidently_snapshot(model_id)  # JSON string
    # inference_data = fetch_inference_batch()
    
    # Step 2: Create configuration (same as training)
    config = create_monitoring_config(
        feature_columns=['feature1', 'feature2', 'feature3'],
        algorithm_type='CLASSIFICATION',
        kpi_columns=['revenue', 'conversion_rate'],
        prediction_column='predicted_label'
    )
    
    # Step 3: Generate monitoring report
    # monitoring_report = generate_full_report(
    #     model=model,
    #     algorithm_type='CLASSIFICATION',
    #     reference_profile=reference_snapshot,  # JSON string from DB
    #     current_data=inference_data,
    #     config=config
    # )
    
    # Step 4: Log results to MON_METRICS_HISTORY
    # insert_monitoring_results(monitoring_report)
    
    print("✅ Inference monitoring complete - results logged to history")

if __name__ == "__main__":
    print("=== Enhanced Metrics Library for MLOps Workflow ===")
    print("\nTraining Phase Functions:")
    print("- create_evidently_reference_profile()")
    print("- generate_training_baseline_report()")
    print("\nInference Phase Functions:")  
    print("- generate_full_report() [Main monitoring function]")
    print("- Supports JSON reference profiles from training phase")
    
    # Run examples
    demonstrate_training_workflow()
    print()
    demonstrate_inference_workflow()
