"""
Updated functions for etcd_analyzer_elt_json2table.py
Extract, Load, Transform module for ETCD Analyzer Performance Data
Main module for converting JSON outputs to table format and generating brief results
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import json
import pandas as pd
from datetime import datetime
import re
from tabulate import tabulate

# Import specialized ELT modules
from .etcd_analyzer_elt_cluster_info import clusterInfoELT
from .etcd_analyzer_elt_utility import utilityELT
from .etcd_analyzer_elt_cluster_status import etcdClusterStatusELT
from .etcd_analyzer_elt_disk_io import diskIOELT
from .etcd_analyzer_elt_general_info import generalInfoELT
from .etcd_analyzer_elt_wal_fsync import diskWalFsyncELT
from .etcd_analyzer_elt_backend_commit import backendCommitELT
from .etcd_analyzer_elt_compact_defrag import compactDefragELT
from .etcd_analyzer_elt_network_io import networkIOELT
from .etcd_analyzer_elt_deep_drive import deepDriveELT
from .etcd_analyzer_elt_bottleneck import bottleneckELT
from .etcd_analyzer_performance_elt_report import etcdReportELT

logger = logging.getLogger(__name__)

class PerformanceDataELT(utilityELT):
    """Main Extract, Load, Transform class for performance data"""    
    def __init__(self):
        super().__init__()
        self.processed_data = {}
        # Initialize specialized ELT modules
        self.cluster_info_elt = clusterInfoELT()
        self.cluster_status_elt = etcdClusterStatusELT()
        self.disk_io_elt = diskIOELT()
        self.wal_fsync_elt = diskWalFsyncELT() 
        self.backend_commit_elt = backendCommitELT()
        self.compact_defrag_elt = compactDefragELT()
        self.general_info_elt = generalInfoELT()
        self.network_io_elt = networkIOELT()
        self.deep_drive_elt = deepDriveELT()
        self.bottleneck_elt = bottleneckELT()
        self.report_elt = etcdReportELT()
        self.logger=logging.getLogger(__name__)

    def extract_json_data(self, results: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Extract relevant data from tool results"""
        try:
            # Normalize input to a dictionary
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON string in extract_json_data: {e}")
                    return {'error': f"Invalid JSON string: {str(e)}", 'raw_data': results}
            
            if not isinstance(results, dict):
                return {'error': 'Input must be a dictionary or valid JSON string', 'raw_data': results}

            # Identify data type first
            data_type = self._identify_data_type(results)
            
            # For cluster_info and etcd_cluster_status, extract the actual data from nested structure
            actual_data = results
            if data_type == 'compact_defrag':
                if 'data' in results and isinstance(results.get('data'), dict):
                    # Extract the nested data for processing, but pass the full structure
                    actual_data = results  # Pass full structure, not just results['data']
                elif 'result' in results and isinstance(results.get('result'), dict) and 'data' in results['result']:
                    actual_data = results['result']

            if data_type in ['cluster_info', 'etcd_cluster_status', 'disk_io']:
                if 'data' in results and isinstance(results.get('data'), dict):
                    # Extract the nested data for processing
                    actual_data = results['data']
                elif 'result' in results and isinstance(results.get('result'), dict) and 'data' in results['result']:
                    actual_data = results['result']['data']

            extracted = {
                'timestamp': results.get('timestamp', actual_data.get('collection_timestamp', datetime.now().isoformat())),
                'data_type': data_type,
                'raw_data': results,
                'structured_data': {}
            }
            
            # Extract structured data using specialized modules
            if data_type == 'cluster_info':
                extracted['structured_data'] = self.cluster_info_elt.extract_cluster_info(actual_data)
            elif data_type == 'etcd_cluster_status':
                extracted['structured_data'] = self.cluster_status_elt.extract_cluster_status(actual_data)
            elif data_type == 'disk_io':
                extracted['structured_data'] = self.disk_io_elt.extract_disk_io(results)
            elif data_type == 'general_info':
                # Normalize nested structure if present
                gen_input = results['data'] if isinstance(results.get('data'), dict) else results
                extracted['structured_data'] = self.general_info_elt.extract_general_info(gen_input)
            elif data_type == 'wal_fsync':
                extracted['structured_data'] = self.wal_fsync_elt.extract_wal_fsync(results) 
            elif data_type == 'backend_commit':
                extracted['structured_data'] = self.backend_commit_elt.extract_backend_commit(actual_data)
            elif data_type == 'compact_defrag':
                extracted['structured_data'] = self.compact_defrag_elt.extract_compact_defrag(actual_data)                
            elif data_type == 'network_io':  # Add this block
                extracted['structured_data'] = self.network_io_elt.extract_network_io(results) 
            elif data_type == 'deep_drive':
                extracted['structured_data'] = self.deep_drive_elt.extract_deep_drive(results)
            elif data_type == 'bottleneck_analysis':
                extracted['structured_data'] = self.bottleneck_elt.extract_bottleneck_analysis(results)
            elif data_type == 'performance_report':
                extracted['structured_data'] = self._extract_performance_report_data(actual_data)
            else:
                extracted['structured_data'] = self._extract_generic_data(results)
            
            return extracted
            
        except Exception as e:
            logger.error(f"Failed to extract JSON data: {e}")
            return {'error': str(e), 'raw_data': results}

    def _identify_data_type(self, data: Dict[str, Any]) -> str:
        """Identify the type of data from tool results"""
        if 'analysis_results' in data and 'performance_report' in data:
            return 'performance_report'
        
        # Check for nested performance report structure
        if 'analysis_results' in data:
            analysis = data['analysis_results']
            if any(key in analysis for key in ['critical_metrics_analysis', 'performance_summary', 'baseline_comparison']):
                return 'performance_report'

        if 'bottleneck_analysis' in data or (
            'disk_io_bottlenecks' in data and 'network_bottlenecks' in data and 
            'root_cause_analysis' in data and 'performance_recommendations' in data
        ):
            return 'bottleneck_analysis'

        # Check for status field indicating bottleneck analysis result
        if data.get('status') == 'success' and any(key in data for key in [
            'disk_io_bottlenecks', 'network_bottlenecks', 'memory_bottlenecks', 'consensus_bottlenecks'
        ]):
            return 'bottleneck_analysis'

        # Check for deep drive data from etcd_analyzer_performance_deepdrive.py
        if 'category' in data and data.get('category') == 'performance_deep_drive':
            return 'deep_drive'
        
        # Check for deep drive data structure
        if 'data' in data and isinstance(data.get('data'), dict):
            data_section = data['data']
            deep_drive_indicators = [
                'general_info_data', 'wal_fsync_data', 'disk_io_data', 
                'network_data', 'backend_commit_data', 'compact_defrag_data'
            ]
            if all(indicator in data_section for indicator in deep_drive_indicators[:3]):  # Check for at least 3 indicators
                return 'deep_drive'
        
        # Check for test_id and analysis structure (deep drive specific)
        if ('test_id' in data and 'analysis' in data and 'summary' in data and 
            'data' in data and isinstance(data.get('data'), dict)):
            data_section = data['data']
            if any(key in data_section for key in ['general_info_data', 'wal_fsync_data']):
                return 'deep_drive'

        # Check for network I/O data from etcd_network_io.py
        if 'tool' in data and data.get('tool') == 'collect_network_io_metrics':
            return 'network_io'
        
        # Check for network I/O data structure
        if 'data' in data and isinstance(data.get('data'), dict) and 'metrics' in data['data']:
            metrics_data = data['data']['metrics']
            # Check if it contains network I/O specific metrics
            network_io_metrics = ['container_network_rx', 'container_network_tx', 'network_peer_round_trip_time_p99',
                                'network_peer_received_bytes', 'network_peer_sent_bytes', 'network_client_grpc_received_bytes',
                                'network_client_grpc_sent_bytes', 'node_network_rx_utilization', 'grpc_active_watch_streams']
            if any(metric in metrics_data for metric in network_io_metrics):
                return 'network_io'

        # Check direct metrics structure for network I/O
        if 'metrics' in data:
            metrics_data = data['metrics']
            network_io_metrics = ['container_network_rx', 'container_network_tx', 'network_peer_round_trip_time_p99',
                                'network_peer_received_bytes', 'network_peer_sent_bytes', 'network_client_grpc_received_bytes',
                                'network_client_grpc_sent_bytes', 'node_network_rx_utilization', 'grpc_active_watch_streams']
            if any(metric in metrics_data for metric in network_io_metrics):
                return 'network_io'

        # Check for category field indicating network_io
        if data.get('category') == 'network_io':
            return 'network_io'

        # Check for compact defrag data from etcd_disk_compact_defrag.py
        if 'tool' in data and data.get('tool') == 'collect_compact_defrag_metrics':
            return 'compact_defrag'
        
        # Check for compact defrag data structure with pods_metrics
        if 'data' in data and isinstance(data.get('data'), dict) and 'pods_metrics' in data['data']:
            metrics_data = data['data']['pods_metrics']
            # Check if it contains compact defrag specific metrics
            compact_defrag_metrics = [
                'debugging_mvcc_db_compacted_keys',
                'debugging_mvcc_db_compaction_duration_sum_delta', 
                'debugging_mvcc_db_compaction_duration_sum',
                'debugging_snapshot_duration',
                'disk_backend_defrag_duration_sum_rate', 
                'disk_backend_defrag_duration_sum',
                'vmstat_pgmajfault_rate', 
                'vmstat_pgmajfault_total'
            ]
            if any(metric in metrics_data for metric in compact_defrag_metrics):
                return 'compact_defrag'

        # Check direct pods_metrics structure for compact defrag
        if 'pods_metrics' in data:
            metrics_data = data['pods_metrics']
            compact_defrag_metrics = [
                'debugging_mvcc_db_compacted_keys',
                'debugging_mvcc_db_compaction_duration_sum_delta', 
                'debugging_mvcc_db_compaction_duration_sum',
                'debugging_snapshot_duration',
                'disk_backend_defrag_duration_sum_rate', 
                'disk_backend_defrag_duration_sum'
            ]
            if any(metric in metrics_data for metric in compact_defrag_metrics):
                return 'compact_defrag'

        # Check for category field indicating compact_defrag
        if data.get('category') == 'disk_compact_defrag':
            return 'compact_defrag'

        if 'tool' in data and data.get('tool') == 'collect_backend_commit_metrics':
            return 'backend_commit'

        # Check for backend commit data structure
        if 'data' in data and isinstance(data.get('data'), dict) and 'metrics' in data['data']:
            metrics_data = data['data']['metrics']
            # Check if it contains backend commit specific metrics
            backend_commit_metrics = ['disk_backend_commit_duration_seconds_p99', 'backend_commit_duration_sum_rate', 
                                    'backend_commit_duration_sum', 'backend_commit_duration_count_rate', 'backend_commit_duration_count']
            if any(metric in metrics_data for metric in backend_commit_metrics):
                return 'backend_commit'

        # Check direct metrics structure for backend commit
        if 'metrics' in data:
            metrics_data = data['metrics']
            backend_commit_metrics = ['disk_backend_commit_duration_seconds_p99', 'backend_commit_duration_sum_rate', 
                                    'backend_commit_duration_sum', 'backend_commit_duration_count_rate', 'backend_commit_duration_count']
            if any(metric in metrics_data for metric in backend_commit_metrics):
                return 'backend_commit'

        # Check for category field indicating backend_commit
        if data.get('category') == 'disk_backend_commit':
            return 'backend_commit'

        # Check for WAL fsync data from etcd_disk_wal_fsync.py
        if 'tool' in data and data.get('tool') == 'collect_wal_fsync_metrics':
            return 'wal_fsync'
        
        # Check for WAL fsync data structure
        if 'data' in data and isinstance(data.get('data'), dict) and 'metrics' in data['data']:
            metrics_data = data['data']['metrics']
            # Check if it contains WAL fsync specific metrics
            wal_fsync_metrics = ['disk_wal_fsync_seconds_duration_p99', 'wal_fsync_duration_seconds_sum_rate', 
                                'wal_fsync_duration_sum', 'wal_fsync_duration_seconds_count_rate', 'wal_fsync_duration_seconds_count']
            if any(metric in metrics_data for metric in wal_fsync_metrics):
                return 'wal_fsync'
        
        # Check direct metrics structure for WAL fsync
        if 'metrics' in data:
            metrics_data = data['metrics']
            wal_fsync_metrics = ['disk_wal_fsync_seconds_duration_p99', 'wal_fsync_duration_seconds_sum_rate', 
                                'wal_fsync_duration_sum', 'wal_fsync_duration_seconds_count_rate', 'wal_fsync_duration_seconds_count']
            if any(metric in metrics_data for metric in wal_fsync_metrics):
                return 'wal_fsync'
        
        # Check for category field indicating wal_fsync
        if data.get('category') == 'disk_wal_fsync':
            return 'wal_fsync'

        # Check for general info data from etcd_general_info.py
        if 'tool' in data and data.get('tool') == 'collect_general_info_metrics':
            return 'general_info'

        # Check for general info data structure
        if 'data' in data and 'pod_metrics' in data['data']:
            return 'general_info'

        # Direct general info structure
        if 'pod_metrics' in data and 'category' in data and data.get('category') == 'general_info':
            return 'general_info'   

        # Check for disk I/O data from etcd_disk_io.py
        if 'tool' in data and data.get('tool') == 'collect_disk_io_metrics':
            return 'disk_io'
        
        # Check for disk I/O data structure
        if 'data' in data and 'metrics' in data['data']:
            metrics_data = data['data']['metrics']
            # Check if it contains disk I/O specific metrics
            disk_io_metrics = ['container_disk_writes', 'node_disk_throughput_read', 
                            'node_disk_throughput_write', 'node_disk_iops_read', 'node_disk_iops_write']
            if any(metric in metrics_data for metric in disk_io_metrics):
                return 'disk_io'
        
        # Check direct metrics structure
        if 'metrics' in data:
            metrics_data = data['metrics']
            disk_io_metrics = ['container_disk_writes', 'node_disk_throughput_read', 
                            'node_disk_throughput_write', 'node_disk_iops_read', 'node_disk_iops_write']
            if any(metric in metrics_data for metric in disk_io_metrics):
                return 'disk_io'
        
        # Check for category field indicating disk_io
        if data.get('category') == 'disk_io':
            return 'disk_io'
        
        # Check for etcd cluster status from etcd_cluster_status.py
        if 'tool' in data and data.get('tool') == 'get_etcd_cluster_status':
            return 'etcd_cluster_status'
        
        # Check for etcd cluster status structure
        if 'data' in data:
            nested_data = data['data']
            if ('etcd_pod' in nested_data and 'cluster_health' in nested_data and 
                'endpoint_status' in nested_data):
                return 'etcd_cluster_status'
        
        # Direct etcd cluster status structure
        if ('etcd_pod' in data and 'cluster_health' in data and 
            'endpoint_status' in data):
            return 'etcd_cluster_status'
        
        # Check for cluster info from ocp_cluster_info.py
        if 'tool' in data and data.get('tool') == 'get_ocp_cluster_info':
            return 'cluster_info'
        
        # Check nested structure for cluster info
        if 'result' in data and isinstance(data.get('result'), dict) and 'data' in data['result']:
            nested_data = data['result']['data']
            if ('cluster_name' in nested_data and 'cluster_version' in nested_data and 
                'master_nodes' in nested_data):
                return 'cluster_info'
        
        # Check for data nested structure (common pattern) for cluster info
        if 'data' in data and isinstance(data.get('data'), dict):
            nested_data = data['data']
            if ('cluster_name' in nested_data and 'cluster_version' in nested_data and 
                ('master_nodes' in nested_data or 'total_nodes' in nested_data)):
                return 'cluster_info'
        
        # Direct cluster info structure
        if ('cluster_name' in data and 'cluster_version' in data and 
            ('master_nodes' in data or 'total_nodes' in data)):
            return 'cluster_info'
        
        # Legacy checks for other data types
        if 'version' in data and 'identity' in data:
            return 'cluster_info'
        elif 'nodes_by_role' in data or 'total_nodes' in data:
            return 'node_info'
        else:
            return 'generic'

    def _extract_generic_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generic data with smart column limiting"""
        structured = {'key_value_pairs': []}
        
        def extract_important_fields(d: Dict[str, Any], max_fields: int = 20) -> List[Tuple[str, Any]]:
            """Extract most important fields from a dictionary"""
            fields = []
            
            # Priority fields (always include if present)
            priority_keys = [
                'name', 'status', 'version', 'timestamp', 'count', 'total',
                'health', 'error', 'message', 'value', 'metric', 'result'
            ]
            
            # Add priority fields first
            for key in priority_keys:
                if key in d:
                    fields.append((key, d[key]))
            
            # Add remaining fields up to limit
            remaining_keys = [k for k in d.keys() if k not in priority_keys]
            for key in remaining_keys:
                if len(fields) < max_fields:
                    fields.append((key, d[key]))
                else:
                    break
            
            return fields
        
        important_fields = extract_important_fields(data)
        
        for key, value in important_fields:
            # Format value for display
            if isinstance(value, (dict, list)):
                if isinstance(value, dict):
                    display_value = f"Dict({len(value)} keys)"
                else:
                    display_value = f"List({len(value)} items)"
            else:
                value_str = str(value)
                display_value = value_str[:100] + '...' if len(value_str) > 100 else value_str
            
            structured['key_value_pairs'].append({
                'Property': key.replace('_', ' ').title(),
                'Value': display_value
            })
        
        return structured

    def _extract_performance_report_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance report data into structured format"""
        try:
            return {
                'test_id': data.get('test_id', 'Unknown'),
                'timestamp': data.get('timestamp', ''),
                'duration': data.get('duration', ''),
                'status': data.get('status', 'unknown'),
                'analysis_results': data.get('analysis_results', {}),
                'performance_report': data.get('performance_report', ''),
                'error': data.get('error', None)
            }
        except Exception as e:
            self.logger.error(f"Error extracting performance report data: {e}")
            return {'error': str(e)}

    def generate_brief_summary(self, structured_data: Dict[str, Any], data_type: str) -> str:
        """Generate a brief textual summary of the data using specialized modules"""
        try:
            if data_type == 'cluster_info':
                return self.cluster_info_elt.summarize_cluster_info(structured_data)
            elif data_type == 'etcd_cluster_status':
                return self.cluster_status_elt.summarize_cluster_status(structured_data)
            elif data_type == 'disk_io':
                return self.disk_io_elt.summarize_disk_io(structured_data)
            elif data_type == 'wal_fsync':
                return self.wal_fsync_elt.summarize_wal_fsync(structured_data)
            elif data_type == 'backend_commit':
                return self.backend_commit_elt.summarize_backend_commit(structured_data)
            elif data_type == 'general_info':
                return self.general_info_elt.summarize_general_info(structured_data)
            elif data_type == 'compact_defrag':
                return self.compact_defrag_elt.summarize_compact_defrag(structured_data)
            elif data_type == 'network_io':  # Add this line
                return self.network_io_elt.summarize_network_io(structured_data) 
            elif data_type == 'deep_drive':
                return self.deep_drive_elt.summarize_deep_drive(structured_data) 
            elif data_type == 'bottleneck_analysis':
                return self.bottleneck_elt.summarize_bottleneck_analysis(structured_data)
            elif data_type == 'performance_report':
                return self._summarize_performance_report(structured_data)
            else:
                return self._summarize_generic(structured_data)
        
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return f"Summary generation failed: {str(e)}"

    def transform_to_dataframes(self, structured_data: Dict[str, Any], data_type: str) -> Dict[str, pd.DataFrame]:
        """Transform structured data into pandas DataFrames using specialized modules"""
        try:
            if data_type == 'cluster_info':
                return self.cluster_info_elt.transform_to_dataframes(structured_data)
            elif data_type == 'etcd_cluster_status':
                return self.cluster_status_elt.transform_to_dataframes(structured_data)
            elif data_type == 'disk_io':
                return self.disk_io_elt.transform_to_dataframes(structured_data)
            elif data_type == 'wal_fsync':
                return self.wal_fsync_elt.transform_to_dataframes(structured_data)
            elif data_type == 'backend_commit':
                return self.backend_commit_elt.transform_to_dataframes(structured_data)
            elif data_type == 'general_info':
                return self.general_info_elt.transform_to_dataframes(structured_data)
            elif data_type == 'compact_defrag':
                return self.compact_defrag_elt.transform_to_dataframes(structured_data) 
            elif data_type == 'network_io':
                logger.info("Processing network_io data type")  # Add this for debugging
                return self.network_io_elt.transform_to_dataframes(structured_data)  
            elif data_type == 'deep_drive':
                return self.deep_drive_elt.transform_to_dataframes(structured_data)
            elif data_type == 'bottleneck_analysis':
                return self.bottleneck_elt.transform_to_dataframes(structured_data)
            elif data_type == 'performance_report':
                return self._transform_performance_report_to_dataframes(structured_data)
            else:
                # Default transformation for other data types
                dataframes = {}
                for key, value in structured_data.items():
                    if isinstance(value, list) and value:
                        df = pd.DataFrame(value)
                        if not df.empty:
                            df = self.limit_dataframe_columns(df, table_name=key)
                            dataframes[key] = df
                return dataframes

        except Exception as e:
            logger.error(f"Failed to transform to DataFrames: {e}")
            return {}

    def _transform_performance_report_to_dataframes(self, data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Transform performance report data to DataFrames"""
        try:
            # Use the existing etcdReportELT to get HTML tables
            html_tables = self.report_elt.process_performance_report(data)
            
            # Convert key information to DataFrames
            dataframes = {}
            
            # Executive summary
            summary_data = [
                {'Property': 'Test ID', 'Value': data.get('test_id', 'Unknown')},
                {'Property': 'Status', 'Value': data.get('status', 'unknown')},
                {'Property': 'Duration', 'Value': data.get('duration', 'Unknown')},
                {'Property': 'Timestamp', 'Value': data.get('timestamp', '')},
            ]
            
            analysis_results = data.get('analysis_results', {})
            if analysis_results:
                # Add critical metrics summary
                critical_analysis = analysis_results.get('critical_metrics_analysis', {})
                if critical_analysis:
                    summary_data.append({
                        'Property': 'Overall Disk Health', 
                        'Value': critical_analysis.get('overall_disk_health', 'unknown')
                    })
                
                # Add performance grade
                baseline = analysis_results.get('baseline_comparison', {})
                if baseline:
                    summary_data.append({
                        'Property': 'Performance Grade',
                        'Value': baseline.get('performance_grade', 'unknown')
                    })
                
                # Add alert count
                alerts = analysis_results.get('alerts', [])
                summary_data.append({
                    'Property': 'Total Alerts',
                    'Value': str(len(alerts))
                })
            
            dataframes['report_summary'] = pd.DataFrame(summary_data)
            
            # Store HTML tables as metadata for later use
            dataframes['_html_tables'] = html_tables
            
            return dataframes
            
        except Exception as e:
            self.logger.error(f"Error transforming performance report: {e}")
            return {}

    def _summarize_generic(self, data: Dict[str, Any]) -> str:
        """Generate generic summary"""
        summary = ["Data Summary:"]
        if 'key_value_pairs' in data:
            summary.append(f"• Total properties: {len(data['key_value_pairs'])}")
            # Show first few important properties
            for item in data['key_value_pairs'][:3]:
                value_preview = str(item['Value'])[:30] + "..." if len(str(item['Value'])) > 30 else str(item['Value'])
                summary.append(f"• {item['Property']}: {value_preview}")
        return " ".join(summary)

    def _summarize_performance_report(self, data: Dict[str, Any]) -> str:
        """Generate summary for performance report"""
        try:
            test_id = data.get('test_id', 'Unknown')
            status = data.get('status', 'unknown')
            duration = data.get('duration', 'Unknown')
            
            analysis_results = data.get('analysis_results', {})
            
            # Get critical health status
            critical_analysis = analysis_results.get('critical_metrics_analysis', {})
            disk_health = critical_analysis.get('overall_disk_health', 'unknown')
            
            # Get performance grade
            baseline = analysis_results.get('baseline_comparison', {})
            performance_grade = baseline.get('performance_grade', 'unknown')
            
            # Count alerts
            alerts = analysis_results.get('alerts', [])
            critical_alerts = len([a for a in alerts if a.get('severity', '').lower() == 'critical'])
            
            summary = [f"Performance Analysis Report: {test_id}"]
            summary.append(f"Analysis Duration: {duration}")
            summary.append(f"Overall Status: {status.title()}")
            summary.append(f"Disk Health: {disk_health.title()}")
            summary.append(f"Performance Grade: {performance_grade.title()}")
            
            if alerts:
                summary.append(f"Active Alerts: {len(alerts)} ({critical_alerts} critical)")
            else:
                summary.append("No active alerts")
            
            return " • ".join(summary)
            
        except Exception as e:
            self.logger.error(f"Error summarizing performance report: {e}")
            return f"Performance report summary error: {str(e)}"

    def generate_html_tables(self, dataframes: Dict[str, pd.DataFrame], data_type: str) -> Dict[str, str]:
        """Generate HTML tables using specialized modules"""
        try:
            if data_type == 'cluster_info':
                html_tables = self.cluster_info_elt.generate_html_tables(dataframes)
            elif data_type == 'etcd_cluster_status':
                html_tables = self.cluster_status_elt.generate_html_tables(dataframes)
            elif data_type == 'disk_io':
                html_tables = self.disk_io_elt.generate_html_tables(dataframes)
            elif data_type == 'wal_fsync':
                html_tables = self.wal_fsync_elt.generate_html_tables(dataframes)
            elif data_type == 'backend_commit':
                html_tables = self.backend_commit_elt.generate_html_tables(dataframes)
            elif data_type == 'general_info':
                html_tables = self.general_info_elt.generate_html_tables(dataframes)
            elif data_type == 'compact_defrag':
                html_tables = self.compact_defrag_elt.generate_html_tables(dataframes)
            elif data_type == 'network_io':
                html_tables = self.network_io_elt.generate_html_tables(dataframes)
            elif data_type == 'deep_drive':
                html_tables = self.deep_drive_elt.generate_html_tables(dataframes)
            elif data_type == 'bottleneck_analysis':
                html_tables = self.bottleneck_elt.generate_html_tables(dataframes)
            elif data_type == 'performance_report':
                # Check if we have pre-generated HTML tables
                if '_html_tables' in dataframes:
                    html_tables = dataframes.pop('_html_tables')  # Remove metadata
                else:
                    # Fallback: generate using report ELT
                    raw_data = {'analysis_results': structured_data.get('analysis_results', {})}
                    html_tables = self.report_elt.process_performance_report(raw_data)
                
                # Add summary table from DataFrame
                if 'report_summary' in dataframes:
                    summary_html = self.create_html_table(dataframes['report_summary'], 'report_summary')
                    html_tables['report_summary'] = summary_html                
            else:
                html_tables = {}

            # Fallback: if specialized generator returned no tables, generate default ones
            if not html_tables:
                # Default HTML table generation
                for name, df in dataframes.items():
                    if not df.empty:
                        html_tables[name] = self.create_html_table(df, name)
                return html_tables
            return html_tables
        except AttributeError as e:
            logger.error(f"Missing method in ELT module: {e}")
            # Fallback to default HTML table generation
            html_tables = {}
            for name, df in dataframes.items():
                if not df.empty:
                    html_tables[name] = self.create_html_table(df, name)
            return html_tables
        except Exception as e:
            logger.error(f"Failed to generate HTML tables: {e}")
            return {}

# Updated main extraction function
def extract_and_transform_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and transform tool results into tables and summaries"""
    try:
        elt = PerformanceDataELT()
        
        # Extract data
        extracted = elt.extract_json_data(results)
        
        if 'error' in extracted:
            return extracted
        
        # Create DataFrames using specialized modules
        dataframes = elt.transform_to_dataframes(extracted['structured_data'], extracted['data_type'])
        if not isinstance(dataframes, dict):
            dataframes = {}
        logger.info(f"dataframes of extract_and_transform_results: {dataframes}")
        # Generate HTML tables using specialized modules
        html_tables = elt.generate_html_tables(dataframes, extracted['data_type'])
        # Fallback: if empty, generate default tables from DataFrames
        if not html_tables:
            html_tables = {}
            for name, df in dataframes.items():
                if hasattr(df, 'empty') and not df.empty:
                    html_tables[name] = elt.create_html_table(df, name)
        logger.info(f"html_tables of extract_and_transform_results: {html_tables}")
        # Generate summary using specialized modules
        summary = elt.generate_brief_summary(extracted['structured_data'], extracted['data_type'])

        logger.info(f"summary of extract_and_transform_results: {summary}")
        return {
            'data_type': extracted['data_type'],
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': extracted['structured_data'],
            'timestamp': extracted['timestamp']
        }
        
    except Exception as e:
        logger.error(f"Failed to extract and transform results: {e}")
        return {'error': str(e)}
 
def json_to_html_table(json_data: Union[Dict[str, Any], str], compact: bool = True, two_column: bool = False) -> str:
    """Convert JSON data to HTML table format with option for 2-column layout"""
    try:
        result = convert_json_to_tables(json_data, "html", compact, two_column)
        
        if 'error' in result:
            return f"<div class='alert alert-danger'>Error: {result['error']}</div>"
        
        html_tables = result.get('html', {})
        if not html_tables:
            return "<div class='alert alert-warning'>No tables generated</div>"
        
        # Generate organized output with metrics highlighting
        output_parts = []

        data_type = result.get('metadata', {}).get('data_type', 'unknown')
        
        # Add data type badge for non-cluster data
        if data_type not in ['cluster_info', 'etcd_cluster_status']:
            type_display = data_type.replace('_', ' ').title()
            if data_type == 'disk_io':
                type_display = 'Disk I/O Performance'
            elif data_type == 'wal_fsync':
                type_display = 'WAL Fsync Performance'
            elif data_type == 'backend_commit':
                type_display = 'Backend Commit Performance'
            elif data_type == 'network_io':
                type_display = 'Network I/O Performance'
            elif data_type == 'deep_drive':
                type_display = 'Performance Deep Drive Analysis'
                
            output_parts.append(
                f"<div class='mb-3'><span class='badge badge-info' style='font-size: 1.0rem; font-weight: 600'>{type_display}</span></div>"
            )

        # Add summary if available
        if result.get('summary'):
            summary_val = result['summary']
            # If summary is HTML, pass through; otherwise format
            if isinstance(summary_val, str) and summary_val.strip().startswith('<'):
                try:
                    util = utilityELT()
                    decoded_summary = util.decode_unicode_escapes(summary_val)
                except Exception:
                    decoded_summary = summary_val
                output_parts.append(f"<div class='alert alert-light'>{decoded_summary}</div>")
            else:
                # Format summary text
                try:
                    util = utilityELT()
                    summary_text = util.decode_unicode_escapes(str(summary_val))
                except Exception:
                    summary_text = str(summary_val)

                def _escape_html(s: str) -> str:
                    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

                # Special formatting for performance summaries
                performance_keywords = ['Performance', 'Analysis', 'Deep Drive']
                if any(keyword in summary_text for keyword in performance_keywords) or 'bullet' in summary_text:
                    parts = [p.strip() for p in summary_text.split('<br>') if p.strip()]
                    if not parts:
                        parts = [p.strip() for p in summary_text.split('•') if p.strip()]
                    
                    if parts:
                        first_line = _escape_html(parts[0])
                        # Bold performance analysis titles
                        for keyword in performance_keywords:
                            if keyword in first_line:
                                first_line = first_line.replace(keyword, f'<strong>{keyword}</strong>')
                        
                        remaining_parts = [_escape_html(p) for p in parts[1:] if p]
                        formatted_summary = first_line
                        if remaining_parts:
                            formatted_summary += '<br>' + '<br>'.join(remaining_parts)
                        output_parts.append(f"<div class='alert alert-light'>{formatted_summary}</div>")
                    else:
                        safe_summary = _escape_html(summary_text)
                        output_parts.append(f"<div class='alert alert-light'>{safe_summary}</div>")
                else:
                    safe_summary = _escape_html(summary_text)
                    output_parts.append(f"<div class='alert alert-light'>{safe_summary}</div>")
        
        # Add tables with priority ordering based on data type
        table_priority = []
        
        if data_type == 'deep_drive':
            table_priority = [
                'test_overview', 'analysis_summary', 'general_info', 'wal_fsync', 
                'disk_io', 'network_io', 'backend_commit', 'compact_defrag'
            ]
        elif data_type == 'network_io':
            table_priority = [
                'metrics_overview', 'container_metrics', 'node_performance', 'grpc_streams'
            ]
        # Priority for disk I/O
        elif data_type == 'disk_io':
            table_priority = [
                'metrics_overview', 'node_performance', 'throughput_metrics', 
                'iops_metrics', 'container_metrics'
            ]             
        # Priority for etcd cluster status
        elif data_type == 'etcd_cluster_status':
            table_priority = [
                'cluster_overview', 'endpoint_status', 'health_status',
                'member_details', 'metrics_summary'
            ]
        # Priority for cluster info
        elif data_type == 'cluster_info':
            table_priority = [
                'cluster_overview', 'resource_metrics', 'node_capacity_metrics', 
                'node_distribution', 'resource_summary', 'cluster_health_status',
                'master_nodes_detail', 'worker_nodes_detail', 'mcp_status_detail'
            ]
        elif data_type == 'wal_fsync':
            table_priority = [
                'metrics_overview', 
                'pod_performance_summary',
                'detailed_fsync_seconds_duration_p99', 
                'detailed_fsync_duration_seconds_sum_rate',
                'detailed_fsync_duration_sum',
                'detailed_fsync_duration_seconds_count_rate', 
                'detailed_fsync_duration_seconds_count'
            ]
        elif data_type == 'compact_defrag':
            table_priority = [
                'metrics_overview', 'pod_performance', 'node_performance',
                'compaction_metrics', 'defragmentation_metrics', 'snapshot_metrics', 'page_fault_metrics'
            ]
        elif data_type == 'backend_commit':
            table_priority = [
                'metrics_overview', 'pod_performance', 'latency_details', 
                'operations_rate', 'cumulative_metrics'
            ]
        elif data_type == 'bottleneck_analysis':
            table_priority = [
                'test_overview', 'bottleneck_summary', 'disk_io_details', 
                'network_details', 'memory_details', 'consensus_details',
                'root_cause_analysis', 'performance_recommendations'
            ]
        elif data_type == 'performance_report':
            table_priority = [
                'report_summary', 'executive_summary', 'wal_fsync_analysis', 
                'backend_commit_analysis', 'cpu_analysis', 'memory_analysis',
                'network_analysis', 'disk_io_analysis', 'baseline_comparison',
                'alerts', 'recommendations'
            ]                  
        else:
            table_priority = []
        
        # Add priority tables first
        added_tables = set()
        for priority_table in table_priority:
            if priority_table in html_tables:
                table_title = priority_table.replace('_', ' ').title()
                
                # Special titles for different data types
                # In json_to_html_table function, add to the title_mapping for general_info:
                if data_type == 'general_info':
                    title_mapping = {
                        'pod_performance': 'Pod Performance Details',
                        'node_performance': 'Node Performance Details',
                        'metrics_overview': 'Metrics Overview',
                        'resource_objects': 'API Server Storage Objects (Top 20)'
                    }
                    table_title = title_mapping.get(priority_table, table_title)                
                elif data_type == 'deep_drive':
                    title_mapping = {
                        'test_overview': 'Test Overview',
                        'analysis_summary': 'Analysis Summary',
                        'general_info': 'General Performance Metrics',
                        'wal_fsync': 'WAL Fsync Performance',
                        'disk_io': 'Disk I/O Performance',
                        'network_io': 'Network I/O Performance',
                        'backend_commit': 'Backend Commit Performance',
                        'compact_defrag': 'Compact & Defrag Performance'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                elif data_type == 'network_io':
                    title_mapping = {
                        'metrics_overview': 'Network Metrics Overview',
                        'container_metrics': 'Container Network Usage',
                        'node_performance': 'Node Network Usage',
                        'grpc_streams': 'gRPC Active Stream'
                    }
                    table_title = title_mapping.get(priority_table, table_title)              
                elif data_type == 'disk_io':
                    title_mapping = {
                        'metrics_overview': 'Performance Metrics Overview',
                        'node_performance': 'Node Performance Details',
                        'throughput_metrics': 'Disk Throughput by Node',
                        'iops_metrics': 'IOPS by Node',
                        'container_metrics': 'Container I/O Metrics'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                elif data_type == 'etcd_cluster_status':
                    title_mapping = {
                        'cluster_overview': 'etcd Cluster Overview',
                        'endpoint_status': 'Endpoint Status (etcdctl format)',
                        'health_status': 'Health Status',
                        'member_details': 'Member Details',
                        'metrics_summary': 'Cluster Metrics'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                elif data_type == 'wal_fsync':
                    title_mapping = {
                        'metrics_overview': 'WAL Fsync Metrics Overview',
                        'pod_performance_summary': 'Pod Performance Summary',
                        'detailed_fsync_seconds_duration_p99': 'P99 Fsync Duration Latency',
                        'detailed_fsync_duration_seconds_sum_rate': 'Fsync Duration Sum Rate',
                        'detailed_fsync_duration_sum': 'Fsync Cumulative Duration',
                        'detailed_fsync_duration_seconds_count_rate': 'Fsync Operation Rate',
                        # 'detailed_fsync_duration_seconds_count_rate': 'Fsync Operation Rate - Detailed Pod Metrics',
                        'detailed_fsync_duration_seconds_count': 'Fsync Total Operations Count'
                    }
                    table_title = title_mapping.get(priority_table, table_title)

                elif data_type == 'backend_commit':
                    title_mapping = {
                        'metrics_overview': 'Backend Commit Metrics Overview',
                        'pod_performance': 'Pod Performance Details',
                        'latency_details': 'P99 Latency Analysis',
                        'operations_rate': 'Commit Operations Rate',
                        'cumulative_metrics': 'Cumulative Statistics'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                elif data_type == 'compact_defrag':
                    table_priority = [
                        'metrics_overview', 'pod_performance', 'node_performance',
                        'compaction_metrics', 'defragmentation_metrics', 'page_fault_metrics'
                    ]

                elif data_type == 'compact_defrag':
                    title_mapping = {
                        'metrics_overview': 'Compact & Defrag Metrics Overview',
                        'pod_performance': 'Pod Performance Details',
                        'node_performance': 'Node Performance Details', 
                        'compaction_metrics': 'DB Compaction Metrics',
                        'defragmentation_metrics': 'Defragmentation Metrics',
                        'snapshot_metrics': 'Snapshot Operation Metrics',
                        'page_fault_metrics': 'Page Fault Metrics'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                elif data_type == 'bottleneck_analysis':
                    title_mapping = {
                        'test_overview': 'Analysis Overview',
                        'bottleneck_summary': 'Bottleneck Summary by Category',
                        'disk_io_details': 'Disk I/O Bottlenecks',
                        'network_details': 'Network Bottlenecks', 
                        'memory_details': 'Memory Bottlenecks',
                        'consensus_details': 'Consensus Bottlenecks',
                        'root_cause_analysis': 'Root Cause Analysis',
                        'performance_recommendations': 'Performance Recommendations'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                elif data_type == 'performance_report':
                    title_mapping = {
                        'report_summary': 'Performance Report Summary',
                        'executive_summary': 'Executive Summary',
                        'wal_fsync_analysis': 'WAL Fsync Performance Analysis',
                        'backend_commit_analysis': 'Backend Commit Performance Analysis',
                        'cpu_analysis': 'CPU Usage Analysis',
                        'memory_analysis': 'Memory Usage Analysis',
                        'network_analysis': 'Network Performance Analysis',
                        'disk_io_analysis': 'Disk I/O Performance Analysis',
                        'baseline_comparison': 'Baseline Performance Comparison',
                        'alerts': 'Active Performance Alerts',
                        'recommendations': 'Performance Recommendations'
                    }
                    table_title = title_mapping.get(priority_table, table_title)
                output_parts.append(f"<h5 class='mt-3'>{table_title}</h5>")
                output_parts.append(html_tables[priority_table])
                added_tables.add(priority_table)
        
        # Add remaining tables
        for table_name, html_table in html_tables.items():
            if table_name not in added_tables:
                table_title = table_name.replace('_', ' ').title()
                # Custom titles for General Info module
                if data_type == 'general_info':
                    if table_name == 'pod_performance':
                        table_title = 'The Usage of General Info Per Pods'
                    elif table_name == 'metrics_overview':
                        table_title = 'The Usage of General Info Overview'
                output_parts.append(f"<h5 class='mt-3'>{table_title}</h5>")
                output_parts.append(html_table)
        
        # Join all parts into final HTML
        final_html = ''.join(output_parts)
        
        # Decode any lingering unicode escapes in the final HTML to ensure symbols render
        try:
            util = utilityELT()
            final_html = util.decode_unicode_escapes(final_html)
        except Exception:
            pass
        # Return as plain string (not JSON encoded) so it can be rendered as HTML
        return final_html
        
    except Exception as e:
        logger.error(f"Error in json_to_html_table: {e}")
        return f"<div class='alert alert-danger'>Error generating HTML table: {str(e)}</div>"

def network_io_json_to_html(json_data: Union[Dict[str, Any], str], compact: bool = True, two_column: bool = False) -> str:
    """Convenience wrapper to render Network I/O JSON via json_to_html_table"""
    return json_to_html_table(json_data, compact=compact, two_column=two_column)

def convert_json_to_tables(json_data: Union[Dict[str, Any], str], 
                          table_format: str = "both",
                          compact: bool = True,
                          two_column: bool = False) -> Dict[str, Union[str, List[List]]]:
    """Convert JSON/dictionary data to table formats with enhanced cluster info support"""
    try:
        # Parse JSON string if needed
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return {
                    'error': f"Invalid JSON string: {str(e)}",
                    'metadata': {'conversion_failed': True}
                }
        else:
            data = json_data
        
        if not isinstance(data, dict):
            return {
                'error': "Input data must be a dictionary or JSON object",
                'metadata': {'conversion_failed': True}
            }
        
        # Use the main ELT processor
        transformed = extract_and_transform_results(data)
        
        if 'error' in transformed:
            return {
                'error': f"Data transformation failed: {transformed['error']}",
                'metadata': {'conversion_failed': True}
            }
        
        result = {
            'metadata': {
                'data_type': transformed['data_type'],
                'timestamp': transformed.get('timestamp'),
                'tables_generated': len(transformed['html_tables']),
                'table_names': list(transformed['html_tables'].keys()),
                'conversion_successful': True,
                'compact_mode': compact,
                'two_column_mode': two_column
            },
            'summary': transformed['summary']
        }
        
        # Add requested formats
        if table_format in ["html", "both"]:
            result['html'] = transformed['html_tables']
        
        if table_format in ["tabular", "both"]:
            # Convert DataFrames to tabular format
            tabular_tables = {}
            for table_name, df in transformed['dataframes'].items():
                if not df.empty:
                    # Apply two-column limitation if requested
                    if two_column and len(df.columns) > 2:
                        df_limited = df.iloc[:, :2]  # Take first 2 columns
                    else:
                        df_limited = df
                    
                    raw_data = [df_limited.columns.tolist()] + df_limited.values.tolist()
                    try:
                        formatted_table = tabulate(
                            df_limited.values.tolist(), 
                            headers=df_limited.columns.tolist(),
                            tablefmt="grid",
                            stralign="left",
                            maxcolwidths=[30] * len(df_limited.columns)
                        )
                        tabular_tables[table_name] = {
                            'raw_data': raw_data,
                            'formatted_string': formatted_table
                        }
                    except Exception as e:
                        logger.warning(f"Failed to format table {table_name}: {e}")
                        tabular_tables[table_name] = {
                            'raw_data': raw_data,
                            'formatted_string': str(df_limited)
                        }
            result['tabular'] = tabular_tables
        
        return result
        
    except Exception as e:
        logger.error(f"Error converting JSON to tables: {e}")
        return {
            'error': str(e),
            'metadata': {'conversion_failed': True}
        }

def process_cluster_info_json(cluster_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process cluster info JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(cluster_data, str):
            cluster_data = json.loads(cluster_data)
        
        cluster_elt = clusterInfoELT()
        
        # Extract cluster info
        structured_data = cluster_elt.extract_cluster_info(cluster_data)
        
        # Transform to DataFrames
        dataframes = cluster_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = cluster_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = cluster_elt.summarize_cluster_info(structured_data)
        
        return {
            'data_type': 'cluster_info',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process cluster info: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_disk_io_json(disk_io_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process disk I/O JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(disk_io_data, str):
            disk_io_data = json.loads(disk_io_data)
        
        disk_io_elt = diskIOELT()
        
        # Extract disk I/O data
        structured_data = disk_io_elt.extract_disk_io(disk_io_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        # Transform to DataFrames
        dataframes = disk_io_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = disk_io_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = disk_io_elt.summarize_disk_io(structured_data)
        
        return {
            'data_type': 'disk_io',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process disk I/O data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_general_info_json(general_info_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process general info JSON data specifically"""
    try:
        if isinstance(general_info_data, str):
            general_info_data = json.loads(general_info_data)
        
        general_info_elt = generalInfoELT()
        structured_data = general_info_elt.extract_general_info(general_info_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        dataframes = general_info_elt.transform_to_dataframes(structured_data)
        html_tables = general_info_elt.generate_html_tables(dataframes)
        summary = general_info_elt.summarize_general_info(structured_data)
        
        return {
            'data_type': 'general_info',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process general info data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

# Add a new convenience function for WAL fsync processing:
def process_wal_fsync_json(wal_fsync_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process WAL fsync JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(wal_fsync_data, str):
            wal_fsync_data = json.loads(wal_fsync_data)
        
        wal_fsync_elt = diskWalFsyncELT()
        
        # Extract WAL fsync data
        structured_data = wal_fsync_elt.extract_wal_fsync(wal_fsync_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        # Transform to DataFrames
        dataframes = wal_fsync_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = wal_fsync_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = wal_fsync_elt.summarize_wal_fsync(structured_data)
        
        return {
            'data_type': 'wal_fsync',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process WAL fsync data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_backend_commit_json(backend_commit_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process backend commit JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(backend_commit_data, str):
            backend_commit_data = json.loads(backend_commit_data)
        
        backend_commit_elt = backendCommitELT()
        
        # Extract backend commit data
        structured_data = backend_commit_elt.extract_backend_commit(backend_commit_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        # Transform to DataFrames
        dataframes = backend_commit_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = backend_commit_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = backend_commit_elt.summarize_backend_commit(structured_data)
        
        return {
            'data_type': 'backend_commit',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process backend commit data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_compact_defrag_json(compact_defrag_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process compact defrag JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(compact_defrag_data, str):
            compact_defrag_data = json.loads(compact_defrag_data)
        
        compact_defrag_elt = compactDefragELT()
        
        # Extract compact defrag data
        structured_data = compact_defrag_elt.extract_compact_defrag(compact_defrag_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        # Transform to DataFrames
        dataframes = compact_defrag_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = compact_defrag_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = compact_defrag_elt.summarize_compact_defrag(structured_data)
        
        return {
            'data_type': 'compact_defrag',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process compact defrag data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_network_io_json(network_io_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process network I/O JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(network_io_data, str):
            network_io_data = json.loads(network_io_data)
        
        network_io_elt = networkIOELT()
        
        # Extract network I/O data
        structured_data = network_io_elt.extract_network_io(network_io_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        # Transform to DataFrames
        dataframes = network_io_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = network_io_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = network_io_elt.summarize_network_io(structured_data)
        
        return {
            'data_type': 'network_io',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        logger.info(f"result of process_network_io_json: {summary}")
    except Exception as e:
        logger.error(f"Failed to process network I/O data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_deep_drive_json(deep_drive_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process deep drive JSON data specifically"""
    try:
        if isinstance(deep_drive_data, str):
            deep_drive_data = json.loads(deep_drive_data)
        
        deep_drive_elt = deepDriveELT()
        structured_data = deep_drive_elt.extract_deep_drive(deep_drive_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        dataframes = deep_drive_elt.transform_to_dataframes(structured_data)
        html_tables = deep_drive_elt.generate_html_tables(dataframes)
        summary = deep_drive_elt.summarize_deep_drive(structured_data)
        
        return {
            'data_type': 'deep_drive',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process deep drive data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

def process_performance_report_json(performance_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process performance report JSON data specifically"""
    try:
        if isinstance(performance_data, str):
            performance_data = json.loads(performance_data)
        
        # Use main ELT processor which now handles performance reports
        result = extract_and_transform_results(performance_data)
        
        return {
            'data_type': 'performance_report',
            'summary': result.get('summary', ''),
            'html_tables': result.get('html_tables', {}),
            'dataframes': result.get('dataframes', {}),
            'structured_data': result.get('structured_data', {}),
            'processing_successful': 'error' not in result
        }
        
    except Exception as e:
        logger.error(f"Failed to process performance report data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }
        
# Add this generic function to etcd_analyzer_elt_json2table.py
def convert_generic_json_to_html_table(json_data: Union[Dict[str, Any], str], 
                                     max_columns: int = 6, 
                                     two_column: bool = False) -> str:
    """
    Convert generic JSON/dictionary data to HTML table format
    This is a lightweight function for simple JSON to table conversion
    """
    try:
        # Parse JSON string if needed
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError as e:
                return f"<div class='alert alert-danger'>Invalid JSON: {str(e)}</div>"
        else:
            data = json_data

        if not isinstance(data, dict):
            return "<div class='alert alert-warning'>Input must be a dictionary or JSON object</div>"

        # Create utility instance
        util = utilityELT()
        
        # Flatten nested structure for simple table
        def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict) and len(str(v)) < 200:  # Only flatten small dicts
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list) and len(v) < 10 and all(isinstance(item, (str, int, float)) for item in v):
                    items.append((new_key, ', '.join(map(str, v))))
                else:
                    # Convert complex values to string representation
                    if isinstance(v, (dict, list)):
                        if isinstance(v, dict):
                            val_str = f"Dict({len(v)} keys)"
                        else:
                            val_str = f"List({len(v)} items)"
                    else:
                        val_str = str(v)[:100] + ('...' if len(str(v)) > 100 else '')
                    items.append((new_key, val_str))
            return dict(items)

        # Flatten the data
        flattened_data = flatten_dict(data)
        
        # Convert to list of dictionaries for DataFrame
        table_data = []
        for key, value in flattened_data.items():
            # Clean up key names
            display_key = key.replace('_', ' ').title()
            display_key = display_key.replace('.', ' > ')  # Show hierarchy
            
            table_data.append({
                'Property': display_key,
                'Value': value
            })
        
        if not table_data:
            return "<div class='alert alert-warning'>No data to display</div>"

        # Create DataFrame
        df = pd.DataFrame(table_data)
        
        # Apply column limits
        if two_column:
            max_columns = 2
        df = util.limit_dataframe_columns(df, max_columns)
        
        # Apply formatting to values
        for idx, row in df.iterrows():
            value = str(row.get('Value', ''))
            # Decode unicode escapes
            try:
                value = util.decode_unicode_escapes(value)
            except:
                pass
            df.at[idx, 'Value'] = value
        
        # Generate HTML table
        html_table = util.create_html_table(df, 'generic_data')
        
        return html_table
        
    except Exception as e:
        logger.error(f"Error in convert_generic_json_to_html_table: {e}")
        return f"<div class='alert alert-danger'>Error generating table: {str(e)}</div>"

def process_bottleneck_analysis_json(bottleneck_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """Process bottleneck analysis JSON data specifically"""
    try:
        # Parse JSON string if needed
        if isinstance(bottleneck_data, str):
            bottleneck_data = json.loads(bottleneck_data)
        
        bottleneck_elt = bottleneckELT()
        
        # Extract bottleneck analysis data
        structured_data = bottleneck_elt.extract_bottleneck_analysis(bottleneck_data)
        
        if 'error' in structured_data:
            return {'error': structured_data['error'], 'processing_successful': False}
        
        # Transform to DataFrames
        dataframes = bottleneck_elt.transform_to_dataframes(structured_data)
        
        # Generate HTML tables
        html_tables = bottleneck_elt.generate_html_tables(dataframes)
        
        # Generate summary
        summary = bottleneck_elt.summarize_bottleneck_analysis(structured_data)
        
        return {
            'data_type': 'bottleneck_analysis',
            'summary': summary,
            'html_tables': html_tables,
            'dataframes': dataframes,
            'structured_data': structured_data,
            'processing_successful': True
        }
        
    except Exception as e:
        logger.error(f"Failed to process bottleneck analysis data: {e}")
        return {
            'error': str(e),
            'processing_successful': False
        }

#  Updated __all__ export list
__all__ = [
    'PerformanceDataELT',
    'extract_and_transform_results',
    'json_to_html_table',
    'network_io_json_to_html',
    'convert_json_to_tables',
    'process_cluster_info_json',
    'process_disk_io_json',
    'process_wal_fsync_json',
    'process_backend_commit_json',
    'process_compact_defrag_json',
    'process_network_io_json',
    'process_deep_drive_json',
    'process_bottleneck_analysis_json',
    'process_performance_report_json',
    'etcdClusterStatusELT',
    'diskIOELT',
    'diskWalFsyncELT',
    'generalInfoELT',
    'backendCommitELT',
    'compactDefragELT',
    'networkIOELT',
    'deepDriveELT',
    'bottleneckELT'
]