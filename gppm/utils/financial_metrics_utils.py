"""
財務指標分析用ユーティリティ関数
"""

from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_metric_time_series(
    y_data: pd.DataFrame,
    entity_id: str,
    fsym_id: str,
    metric_name: str,
    ppm_manager_cls=None,
    pred_length: int = 0,
    width: int = 800,
    height: int = 400,
) -> go.Figure:
    dates = pd.to_datetime(y_data.T.index, format='%Y%m')
    company_name = ""
    if ppm_manager_cls and hasattr(ppm_manager_cls, 'id') and hasattr(ppm_manager_cls.id, 'map_ja'):
        try:
            company_name = ppm_manager_cls.id.map_ja.query(f"FACTSET_ENTITY_ID == '{entity_id}'")['NAME'].iloc[0]
        except (IndexError, KeyError):
            company_name = entity_id

    y_values = y_data.T.loc[:, fsym_id]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_values, mode='lines+markers', name=metric_name, line=dict(width=2), marker=dict(size=6)))
    title = f'企業名: {company_name}' if company_name else f'企業ID: {entity_id}'
    fig.update_layout(title=title, xaxis_title='期間', yaxis_title=metric_name, width=width, height=height, hovermode='x unified')
    fig.update_yaxes(tickformat='.1%')
    return fig


def plot_metric_distribution(
    metric_data: pd.DataFrame,
    metric_column: str,
    width: int = 1200,
    height: int = 600,
    whis: float = 1.5,
) -> None:
    df = metric_data.copy()
    df['年'] = df['FTERM_2'].astype(str).str[:4].astype(int)
    df['月'] = df['FTERM_2'].astype(str).str[-2:].astype(int)
    df.sort_values(by="FTERM_2", inplace=True)
    years = sorted(df['年'].unique())
    all_months = sorted(df['月'].unique())
    cols = len(years)
    subplot_titles = [f'{year}年' for year in years]
    fig = make_subplots(rows=1, cols=cols, subplot_titles=subplot_titles, shared_yaxes=True)
    for i, year in enumerate(years, 1):
        year_data = df[df['年'] == year]
        for month in all_months:
            month_data = year_data[year_data['月'] == month]
            if len(month_data) > 0:
                s = month_data[metric_column]
                q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - whis * iqr
                upper_bound = q3 + whis * iqr
                whisker_min = s[s >= lower_bound].min()
                whisker_max = s[s <= upper_bound].max()
                fig.add_trace(
                    go.Box(q1=[q1], median=[med], q3=[q3], lowerfence=[whisker_min], upperfence=[whisker_max], name=f'{month}月', x=[f'{month}月'], boxpoints=False, marker_color='rgba(0,0,0,0.6)', line_color='rgba(0,0,0,0.8)', showlegend=False),
                    row=1,
                    col=i,
                )
            else:
                fig.add_trace(go.Scatter(x=[f'{month}月'], y=[None], mode='markers', marker=dict(opacity=0), showlegend=False, hoverinfo='skip'), row=1, col=i)
    fig.update_layout(title=f'{metric_column} 分布（年別・月別ボックスプロット、外れ値除外: IQR×{whis}）', width=width, height=height)
    for i in range(1, len(years) + 1):
        fig.update_yaxes(tickformat='.1%', hoverformat='.1%', gridcolor='lightgray', row=1, col=i)
        fig.update_xaxes(gridcolor='lightgray', row=1, col=i)
    fig.update_yaxes(title_text=metric_column, row=1, col=1)
    fig.show()


def plot_metric_histogram(metric_data: pd.DataFrame, metric_column: str, bins: int = 50, width: int = 800, height: int = 500) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=metric_data[metric_column].dropna(), nbinsx=bins, name=metric_column, opacity=0.7))
    fig.update_layout(title=f'{metric_column} 分布', xaxis_title=metric_column, yaxis_title='頻度', width=width, height=height)
    return fig


def calculate_metric_statistics(metric_data: pd.DataFrame, metric_columns: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    if metric_columns is None:
        metric_columns = metric_data.select_dtypes(include=[np.number]).columns.tolist()
    results: Dict[str, Dict[str, float]] = {}
    for col in metric_columns:
        if col in metric_data.columns:
            values = metric_data[col].dropna()
            if len(values) > 0:
                results[col] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                    "count": len(values),
                }
    return results


def compare_metrics(data_dict: Dict[str, pd.DataFrame], metric_columns: Dict[str, str], entity_mapping: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    stats_dict: Dict[str, Dict[str, float]] = {}
    for key, data in data_dict.items():
        if key in metric_columns:
            col = metric_columns[key]
            stats = calculate_metric_statistics(data, [col])
            if col in stats:
                stats_dict[key] = stats[col]
    comparison = pd.DataFrame(stats_dict)
    return comparison


def create_metrics_comparison_plot(
    data: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    color_column: Optional[str] = None,
    title: Optional[str] = None,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    if title is None:
        title = f'{y_metric} vs {x_metric}'
    fig = px.scatter(data, x=x_metric, y=y_metric, color=color_column, title=title, width=width, height=height, opacity=0.6)
    if x_metric == y_metric or ('ROIC' in x_metric and 'WACC' in y_metric) or ('WACC' in x_metric and 'ROIC' in y_metric):
        min_val = float(min(data[x_metric].min(), data[y_metric].min()))
        max_val = float(max(data[x_metric].max(), data[y_metric].max()))
        fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", width=2, dash="dash"))
    return fig


def create_time_series_comparison(
    data_dict: Dict[str, pd.DataFrame],
    metric_columns: Dict[str, str],
    title: str = "Financial Metrics Time Series",
    width: int = 1000,
    height: int = 600,
) -> go.Figure:
    fig = go.Figure()
    for key, data in data_dict.items():
        if key in metric_columns:
            col = metric_columns[key]
            if col in data.columns and 'FTERM_2' in data.columns:
                time_series = data.groupby('FTERM_2')[col].mean().reset_index()
                fig.add_trace(go.Scatter(x=time_series['FTERM_2'], y=time_series[col], mode='lines+markers', name=f'{key} - {col}', line=dict(width=2)))
    fig.update_layout(title=title, xaxis_title='期間', yaxis_title='値', width=width, height=height, hovermode='x unified')
    return fig


def create_correlation_heatmap(data: pd.DataFrame, metrics: List[str], title: str = "財務指標相関マトリックス", width: int = 600, height: int = 500) -> go.Figure:
    correlation_matrix = data[metrics].corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix.values, x=correlation_matrix.columns, y=correlation_matrix.columns, colorscale='RdBu', zmid=0, text=correlation_matrix.round(3).values, texttemplate="%{text}", textfont={"size": 10}, hoverongaps=False))
    fig.update_layout(title=title, width=width, height=height)
    return fig


