"""
General Info ELT module for ETCD Analyzer
Extract, Load, Transform module for etcd general info performance data
"""

import logging
from typing import Dict, Any, List, Union
import pandas as pd
from .etcd_analyzer_elt_utility import utilityELT
from config.etcd_config import get_config

logger = logging.getLogger(__name__)

class generalInfoELT(utilityELT):
    """Extract, Load, Transform class for etcd general info data"""
    
    def __init__(self):
        super().__init__()
        # Load metrics config to map metric names to display titles/units
        try:
            self.config = get_config()
        except Exception:
            self.config = None
    
    def extract_general_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract general info data from tool results"""
        try:
            # Handle nested data structure
            if 'data' in data and isinstance(data['data'], dict):
                actual_data = data['data']
            else:
                actual_data = data
            
            if 'error' in actual_data:
                return {'error': actual_data['error']}
            
            pod_metrics = actual_data.get('pod_metrics', {})
            
            # Extract metrics overview
            metrics_overview = []
            pod_performance = []
            node_performance = []  # Add this for node-level metrics
            resource_objects = []  # Add this for apiserver objects
            
            # Process each metric
            for metric_name, metric_data in pod_metrics.items():
                if isinstance(metric_data, dict):
                    # Handle apiserver_storage_objects_max_top20
                    if metric_name == 'apiserver_storage_objects_max_top20':
                        resources = metric_data.get('resources', [])
                        cfg_title = None
                        cfg_unit = None
                        try:
                            if self.config:
                                cfg = self.config.get_metric_by_name(metric_name)
                                if cfg:
                                    cfg_title = cfg.get('title')
                                    cfg_unit = cfg.get('unit')
                        except Exception:
                            pass
                        
                        display_title = (
                            metric_data.get('title')
                            or cfg_title
                            or 'API Server Storage Objects (Top 20)'
                        )
                        
                        for resource in resources:
                            resource_objects.append({
                                'Resource Name': resource.get('resource_name', '').replace('.', ' '),
                                'Max Objects': resource.get('max_value', 0)
                            })
                        continue
                    
                    # Handle cpu_io_utilization_iowait (node-level metric)
                    if metric_name == 'cpu_io_utilization_iowait':
                        nodes = metric_data.get('nodes', {})
                        cfg_title = None
                        cfg_unit = None
                        try:
                            if self.config:
                                cfg = self.config.get_metric_by_name(metric_name)
                                if cfg:
                                    cfg_title = cfg.get('title')
                                    cfg_unit = cfg.get('unit')
                        except Exception:
                            pass
                        
                        display_title = (
                            metric_data.get('title')
                            or cfg_title
                            or 'CPU I/O Wait Utilization'
                        )
                        unit = metric_data.get('unit', cfg_unit or 'percent')
                        
                        # Calculate cluster-wide statistics for nodes
                        all_avg_values = [node_data.get('avg', 0) for node_data in nodes.values() if 'avg' in node_data]
                        all_max_values = [node_data.get('max', 0) for node_data in nodes.values() if 'max' in node_data]
                        
                        if all_avg_values:
                            cluster_avg = sum(all_avg_values) / len(all_avg_values)
                            cluster_max = max(all_max_values) if all_max_values else 0
                            
                            formatted_avg = self._format_metric_value(cluster_avg, unit)
                            formatted_max = self._format_metric_value(cluster_max, unit)
                            
                            metrics_overview.append({
                                'Metric': display_title,
                                'Unit': unit,
                                'Cluster Avg': formatted_avg,
                                'Cluster Max': formatted_max,
                                'Nodes Count': len(nodes)
                            })
                        
                        # Create detailed node performance data
                        for node_name, node_data in nodes.items():
                            formatted_avg = self._format_metric_value(node_data.get('avg', 0), unit)
                            formatted_max = self._format_metric_value(node_data.get('max', 0), unit)
                            
                            node_performance.append({
                                'Node': self.truncate_node_name(node_name),
                                'Metric': display_title,
                                'Avg': formatted_avg,
                                'Max': formatted_max,
                                'Count': node_data.get('count', 0)
                            })
                        continue
                    
                    # Handle pod-level metrics (existing code)
                    if 'pods' in metric_data:
                        # Prefer title/unit from config if available
                        cfg_title = None
                        cfg_unit = None
                        try:
                            if self.config:
                                cfg = self.config.get_metric_by_name(metric_name)
                                if cfg:
                                    cfg_title = cfg.get('title')
                                    cfg_unit = cfg.get('unit')
                        except Exception:
                            pass
                        # Determine display title and unit with sensible fallbacks
                        display_title = (
                            metric_data.get('title')
                            or cfg_title
                            or metric_name.replace('_', ' ').title()
                        )
                        unit = metric_data.get('unit', cfg_unit or '')
                        pods = metric_data.get('pods', {})
                        
                        # Calculate cluster-wide statistics
                        all_avg_values = [pod_data.get('avg', 0) for pod_data in pods.values() if 'avg' in pod_data]
                        all_max_values = [pod_data.get('max', 0) for pod_data in pods.values() if 'max' in pod_data]
                        
                        if all_avg_values:
                            cluster_avg = sum(all_avg_values) / len(all_avg_values)
                            cluster_max = max(all_max_values) if all_max_values else 0
                            
                            # Format values with appropriate units
                            formatted_avg = self._format_metric_value(cluster_avg, unit)
                            formatted_max = self._format_metric_value(cluster_max, unit)
                            
                            metrics_overview.append({
                                'Metric': display_title,
                                'Unit': unit,
                                'Cluster Avg': formatted_avg,
                                'Cluster Max': formatted_max,
                                'Pods Count': len(pods)
                            })
                        
                        # Create detailed pod performance data
                        for pod_name, pod_data in pods.items():
                            formatted_avg = self._format_metric_value(pod_data.get('avg', 0), unit)
                            formatted_max = self._format_metric_value(pod_data.get('max', 0), unit)
                            
                            pod_performance.append({
                                'Pod': self.truncate_text(pod_name, max_length=60),
                                'Node': self.truncate_node_name(pod_data.get('node', 'unknown')),
                                'Metric': display_title,
                                'Avg': formatted_avg,
                                'Max': formatted_max,
                                'Count': pod_data.get('count', 0)
                            })
            
            # Create high-level cluster summary
            cluster_summary = []
            total_pods = len(set(pod_data.get('node', '') for metric_data in pod_metrics.values() 
                            if isinstance(metric_data, dict) and 'pods' in metric_data 
                            for pod_data in metric_data['pods'].values()))
            
            cluster_summary.append({'Property': 'Collection Timestamp', 'Value': actual_data.get('timestamp', 'N/A')})
            cluster_summary.append({'Property': 'Duration', 'Value': actual_data.get('duration', 'N/A')})
            cluster_summary.append({'Property': 'Total etcd Pods', 'Value': str(total_pods)})
            cluster_summary.append({'Property': 'Metrics Collected', 'Value': str(len(pod_metrics))})
            
            return {
                'cluster_summary': cluster_summary,
                'metrics_overview': metrics_overview,
                'pod_performance': pod_performance,
                'node_performance': node_performance,  # Add this
                'resource_objects': resource_objects,  # Add this
                'timestamp': actual_data.get('timestamp'),
                'duration': actual_data.get('duration', '1h'),
                'category': actual_data.get('category', 'general_info')
            }
            
        except Exception as e:
            logger.error(f"Failed to extract general info data: {e}")
            return {'error': str(e)}

    def _format_metric_value(self, value: float, unit: str) -> str:
        """Format metric value with appropriate units and precision"""
        try:
            if unit == 'percent':
                return f"{value:.2f}%"
            elif unit == 'MB':
                return f"{value:.1f} MB"
            elif unit == 'seconds':
                if value < 0.1:
                    return f"{value*1000:.1f} ms"
                else:
                    return f"{value:.3f} s"
            elif unit == 'per_second':
                return f"{value:.2f}/s"
            elif unit == 'count':
                if value > 1000000:
                    return f"{value/1000000:.1f}M"
                elif value > 1000:
                    return f"{value/1000:.1f}K"
                else:
                    return f"{value:.0f}"
            elif unit == 'per_day':
                return f"{value:.1f}/day"
            elif unit == 'boolean':
                return "Yes" if value == 1 else "No"
            else:
                return f"{value:.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    def transform_to_dataframes(self, structured_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Transform structured data into pandas DataFrames"""
        try:
            dataframes = {}
            
            # Create DataFrames for each data type
            for key, data in structured_data.items():
                if key in ['timestamp', 'duration', 'category']:
                    continue
                
                if isinstance(data, list) and data:
                    df = pd.DataFrame(data)
                    if not df.empty:
                        # For pod_performance: remove Node column and order columns as requested
                        if key == 'pod_performance':
                            desired_order = ['Metric', 'Pod', 'Avg', 'Max', 'Count']
                            existing_cols = [c for c in desired_order if c in df.columns]
                            if existing_cols:
                                df = df[existing_cols]
                            dataframes[key] = df
                        # For node_performance: order columns
                        elif key == 'node_performance':
                            desired_order = ['Metric', 'Node', 'Avg', 'Max', 'Count']
                            existing_cols = [c for c in desired_order if c in df.columns]
                            if existing_cols:
                                df = df[existing_cols]
                            dataframes[key] = df
                        # For resource_objects: no changes needed
                        elif key == 'resource_objects':
                            dataframes[key] = df
                        elif key == 'metrics_overview':
                            # Move 'Unit' after 'Cluster Max'
                            desired_order = ['Metric', 'Cluster Avg', 'Cluster Max', 'Unit', 'Pods Count', 'Nodes Count']
                            existing_cols = [c for c in desired_order if c in df.columns]
                            if existing_cols:
                                df = df[existing_cols]
                            dataframes[key] = df
                        else:
                            # Apply column limiting for other tables
                            df_limited = self.limit_dataframe_columns(df, table_name=key)
                            dataframes[key] = df_limited
            
            return dataframes
            
        except Exception as e:
            logger.error(f"Failed to transform general info data to DataFrames: {e}")
            return {} 

    def generate_html_tables(self, dataframes: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Generate HTML tables with highlighting for general info metrics"""
        try:
            html_tables = {}
            
            for table_name, df in dataframes.items():
                if df.empty:
                    continue
                
                # Apply highlighting for different table types
                if table_name == 'metrics_overview':
                    df_highlighted = self._highlight_metrics_overview(df.copy())
                elif table_name == 'pod_performance':
                    df_highlighted = self._highlight_pod_performance(df.copy())
                elif table_name == 'node_performance':
                    df_highlighted = self._highlight_node_performance(df.copy())
                elif table_name == 'resource_objects':
                    df_highlighted = self._highlight_resource_objects(df.copy())
                else:
                    df_highlighted = df.copy()
                
                html_tables[table_name] = self.create_html_table(df_highlighted, table_name)
            
            return html_tables
            
        except Exception as e:
            logger.error(f"Failed to generate HTML tables for general info: {e}")
            return {}

    def _highlight_metrics_overview(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add highlighting to metrics overview table"""
        if 'Cluster Avg' in df.columns:
            # Define critical thresholds for different metrics
            critical_thresholds = {
                'cpu_usage': {'critical': 80, 'warning': 60},
                'memory_usage': {'critical': 85, 'warning': 70},
                'db_space_used_percent': {'critical': 80, 'warning': 60},
                'proposal_failure_rate': {'critical': 0.1, 'warning': 0.01},
                'slow_applies': {'critical': 0.1, 'warning': 0.05}
            }
            
            for idx, row in df.iterrows():
                metric_name = row.get('Metric', '').lower().replace(' ', '_')
                if metric_name in critical_thresholds:
                    thresholds = critical_thresholds[metric_name]
                    avg_value = self.extract_numeric_value(row.get('Cluster Avg', '0'))
                    max_value = self.extract_numeric_value(row.get('Cluster Max', '0'))
                    
                    df.at[idx, 'Cluster Avg'] = self.highlight_general_info_values(
                        avg_value, metric_name, row.get('Unit', '')
                    )
                    df.at[idx, 'Cluster Max'] = self.highlight_general_info_values(
                        max_value, metric_name, row.get('Unit', '')
                    )
        
        return df
    
    def _highlight_pod_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add highlighting to pod performance table"""
        if 'Avg' in df.columns and 'Max' in df.columns and 'Metric' in df.columns:
            # Group by metric and identify top performers
            for metric_name in df['Metric'].unique():
                metric_rows = df[df['Metric'] == metric_name]
                
                if len(metric_rows) > 1:
                    # Find top values for highlighting
                    avg_values = []
                    for idx, row in metric_rows.iterrows():
                        avg_val = self.extract_numeric_value(row.get('Avg', '0'))
                        avg_values.append((idx, avg_val))
                    
                    if avg_values:
                        # Sort and get top value index
                        avg_values.sort(key=lambda x: x[1], reverse=True)
                        top_idx = avg_values[0][0]
                        
                        # Highlight top performer
                        current_avg = df.at[top_idx, 'Avg']
                        df.at[top_idx, 'Avg'] = self.highlight_general_info_values(
                            avg_values[0][1], metric_name.lower().replace(' ', '_'), '', is_top=True
                        )
        
        return df

    def _highlight_node_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add highlighting to node performance table"""
        if 'Avg' in df.columns and 'Max' in df.columns and 'Metric' in df.columns:
            # Group by metric and identify top performers
            for metric_name in df['Metric'].unique():
                metric_rows = df[df['Metric'] == metric_name]
                
                if len(metric_rows) > 1:
                    # Find top values for highlighting
                    avg_values = []
                    for idx, row in metric_rows.iterrows():
                        avg_val = self.extract_numeric_value(row.get('Avg', '0'))
                        avg_values.append((idx, avg_val))
                    
                    if avg_values:
                        # Sort and get top value index
                        avg_values.sort(key=lambda x: x[1], reverse=True)
                        top_idx = avg_values[0][0]
                        
                        # Highlight top performer
                        current_avg = df.at[top_idx, 'Avg']
                        df.at[top_idx, 'Avg'] = self.highlight_general_info_values(
                            avg_values[0][1], metric_name.lower().replace(' ', '_'), '', is_top=True
                        )
        
        return df

    def _highlight_resource_objects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add highlighting to resource objects table (top 20)"""
        if 'Max Objects' in df.columns:
            # Find top 3 resources for highlighting
            max_values = []
            for idx, row in df.iterrows():
                max_val = row.get('Max Objects', 0)
                max_values.append((idx, max_val))
            
            if max_values:
                # Sort and get top 3
                max_values.sort(key=lambda x: x[1], reverse=True)
                top_3_indices = [v[0] for v in max_values[:3]]
                
                # Highlight top 3
                for rank, idx in enumerate(top_3_indices):
                    current_val = df.at[idx, 'Max Objects']
                    if rank == 0:  # Top 1
                        df.at[idx, 'Max Objects'] = f'<span class="text-primary font-weight-bold bg-light px-1">🏆 {current_val}</span>'
                    elif rank == 1:  # Top 2
                        df.at[idx, 'Max Objects'] = f'<span class="text-info font-weight-bold">⭐ {current_val}</span>'
                    else:  # Top 3
                        df.at[idx, 'Max Objects'] = f'<span class="text-success font-weight-bold">{current_val}</span>'
        
        return df

    def summarize_general_info(self, structured_data: Dict[str, Any]) -> str:
        """Generate a brief summary of general info data"""
        try:
            summary_parts = []
            
            # Basic cluster info
            cluster_summary = structured_data.get('cluster_summary', [])
            pod_count = 0
            metrics_count = 0
            
            for item in cluster_summary:
                if item.get('Property') == 'Total etcd Pods':
                    pod_count = int(item.get('Value', '0'))
                elif item.get('Property') == 'Metrics Collected':
                    metrics_count = int(item.get('Value', '0'))
            
            summary_parts.append(f"etcd General Info: {pod_count} pods, {metrics_count} metrics collected")
            
            # Highlight key metrics
            metrics_overview = structured_data.get('metrics_overview', [])
            critical_metrics = []
            
            for metric in metrics_overview:
                metric_name = metric.get('Metric', '')
                if 'cpu' in metric_name.lower() or 'memory' in metric_name.lower():
                    avg_val = self.extract_numeric_value(metric.get('Cluster Avg', '0'))
                    if avg_val > 50:  # Arbitrary threshold for demo
                        critical_metrics.append(f"{metric_name}: {metric.get('Cluster Avg', 'N/A')}")
            
            if critical_metrics:
                summary_parts.append(f"High usage metrics: {', '.join(critical_metrics[:2])}")
            else:
                summary_parts.append("All metrics within normal ranges")
            
            return " • ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate general info summary: {e}")
            return f"General info summary generation failed: {str(e)}"