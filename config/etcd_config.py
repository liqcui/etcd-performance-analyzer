"""
etcd Configuration Module
Handles configuration loading and management for etcd monitoring
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path


class ETCDConfig:
    """Configuration manager for etcd monitoring"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or self._get_default_config_path()
        self.metrics_config = {}
        self.raw_metrics = []
        self.load_config()
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path"""
        current_dir = Path(__file__).parent
        config_file = current_dir / "metrics-etcd.yml"
        
        # Try different possible locations
        possible_paths = [
            str(config_file),
            str(current_dir.parent / "config" / "metrics-etcd.yml"),
            str(current_dir.parent / "metrics-etcd.yml"),
            "./config/metrics-etcd.yml",
            "./metrics-etcd.yml",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.logger.info(f"Found config file at: {path}")
                return path
        
        self.logger.error(f"Config file not found in any of: {possible_paths}")
        raise FileNotFoundError(f"metrics-etcd.yml not found in any of the expected locations: {possible_paths}")
    
    def load_config(self) -> bool:
        """Load metrics configuration from YAML file"""
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"Configuration file not found: {self.config_path}")
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'metrics' not in config_data:
                self.logger.error("No metrics configuration found in file")
                raise ValueError("No metrics configuration found in file")
            
            self.raw_metrics = config_data['metrics']
            self._process_metrics()
            
            self.logger.info(f"Loaded {len(self.raw_metrics)} metrics from configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            raise
    
    def _process_metrics(self):
        """Process and validate metrics configuration"""
        processed_metrics = {}
        
        for metric in self.raw_metrics:
            if not all(key in metric for key in ['name', 'expr', 'category']):
                self.logger.warning(f"Invalid metric configuration: {metric}")
                continue
            
            # Process variables in expression
            processed_expr = self._process_expression_variables(metric['expr'])
            metric['expr'] = processed_expr
            
            # Organize by category
            category = metric['category']
            if category not in processed_metrics:
                processed_metrics[category] = []
            
            processed_metrics[category].append(metric)
        
        self.metrics_config = processed_metrics
        
        # Log processed metrics
        self.logger.info("Processed metrics by category:")
        for category, metrics in processed_metrics.items():
            self.logger.info(f"  {category}: {len(metrics)} metrics")
            for metric in metrics:
                self.logger.debug(f"    - {metric['name']}: {metric['expr']}")
    
    def _process_expression_variables(self, expr: str) -> str:
        """Process and replace variables in PromQL expressions"""
        # Replace common variable patterns
        replacements = {
            r'\$1': '$1',  # Keep regex captures
            r'\${1}': '$1',
            r'\$\{([^}]+)\}': r'$\1',  # Remove extra braces
        }
        
        processed_expr = expr
        for pattern, replacement in replacements.items():
            import re
            processed_expr = re.sub(pattern, replacement, processed_expr)
        
        return processed_expr
    
    def get_metrics_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get metrics for a specific category"""
        return self.metrics_config.get(category, [])
    
    def get_all_categories(self) -> List[str]:
        """Get all available metric categories"""
        return list(self.metrics_config.keys())
    
    def get_metric_by_name(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get specific metric by name"""
        self.logger.debug(f"Searching for metric: {metric_name}")
        
        # Handle metric names with prefixes (network_io_xxx -> xxx)
        clean_name = metric_name
        if metric_name.startswith('network_io_'):
            clean_name = metric_name[11:]  # Remove 'network_io_' prefix
        
        for category_name, category_metrics in self.metrics_config.items():
            self.logger.debug(f"Checking category {category_name} with {len(category_metrics)} metrics")
            for metric in category_metrics:
                # Check both full name and clean name
                if metric['name'] == metric_name or metric['name'] == clean_name:
                    self.logger.debug(f"Found metric {metric_name} in category {category_name}")
                    return metric
                # Also check for metric names with network_io_ prefix in config
                if metric['name'].startswith('network_io_') and metric['name'] == f"network_io_{clean_name}":
                    self.logger.debug(f"Found metric {metric_name} with prefix in category {category_name}")
                    return metric
        
        self.logger.warning(f"Metric {metric_name} not found in configuration")
        self.logger.debug(f"Available metrics: {[m['name'] for cat in self.metrics_config.values() for m in cat]}")
        return None
    
    def get_all_metric_names(self) -> List[str]:
        """Get list of all metric names"""
        names = []
        for category_metrics in self.metrics_config.values():
            for metric in category_metrics:
                names.append(metric['name'])
        return names
    
    def get_category_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about each category"""
        return {
            'general_info': {
                'description': 'General etcd cluster information and health metrics',
                'collector_class': 'GeneralInfoCollector',
                'file': 'etcd_general_info.py'
            },
            'disk_compact_defrag': {
                'description': 'Database compaction and defragmentation metrics',
                'collector_class': 'CompactDefragCollector', 
                'file': 'etcd_disk_compact_defrag.py'
            },
            'disk_wal_fsync': {
                'description': 'Write-Ahead Log fsync performance metrics',
                'collector_class': 'DiskWALFsyncCollector',
                'file': 'etcd_disk_wal_fsync.py'
            },
            'disk_backend_commit': {
                'description': 'Backend commit operation performance metrics',
                'collector_class': 'DiskBackendCommitCollector',
                'file': 'etcd_disk_backend_commit.py'
            },
            'network_io': {
                'description': 'Network I/O and peer communication metrics',
                'collector_class': 'NetworkIOCollector',
                'file': 'etcd_network_io.py'
            },
            'disk_io': {
                'description': 'Disk I/O performance and throughput metrics',
                'collector_class': 'DiskIOCollector', 
                'file': 'etcd_disk_io.py'
            }
        }
    
    def get_network_io_metrics(self) -> Dict[str, List[str]]:
        """Get network_io metrics grouped by type"""
        network_metrics = self.get_metrics_by_category('network_io')
        
        # Group metrics by function type
        # container_metrics = []
        pods_metrics = []
        node_metrics = []
        cluster_metrics = []
        
        for metric in network_metrics:
            name = metric['name']
            
            # Container/Pod level metrics - check with and without network_io_ prefix
            clean_name = name.replace('network_io_', '') if name.startswith('network_io_') else name
            if any(x in clean_name for x in ['container_network', 'peer2peer_latency', 'peer_received', 'peer_sent', 'client_grpc']):
                pods_metrics.append(clean_name)
                # container_metrics.append(clean_name)
            # Node level metrics  
            elif any(x in clean_name for x in ['node_network']):
                node_metrics.append(clean_name)
            # Cluster level metrics
            elif any(x in clean_name for x in ['grpc_active']):
                cluster_metrics.append(clean_name)
        
        return {
            # 'container_metrics': container_metrics,
            'pods_metrics': pods_metrics,
            'node_metrics': node_metrics,
            'cluster_metrics': cluster_metrics
        }

    def get_disk_io_metrics(self) -> Dict[str, List[str]]:
        """Get disk_io metrics grouped by type"""
        disk_metrics = self.get_metrics_by_category('disk_io')
        
        # Group metrics by function type
        container_metrics = []
        node_metrics = []
        timing_metrics = []
        
        for metric in disk_metrics:
            name = metric['name']
            
            # Container/Pod level metrics
            clean_name = name.replace('disk_io_', '') if name.startswith('disk_io_') else name
            if 'container_disk_writes' in clean_name:
                container_metrics.append(clean_name)
            # Node level throughput and IOPS metrics  
            elif any(x in clean_name for x in ['node_disk_throughput', 'node_disk_iops']):
                node_metrics.append(clean_name)
            # Timing metrics
            elif any(x in clean_name for x in ['read_time_seconds', 'writes_time_seconds', 'io_time_seconds']):
                timing_metrics.append(clean_name)
        
        return {
            'container_metrics': container_metrics,
            'node_metrics': node_metrics, 
            'timing_metrics': timing_metrics
        }

    def validate_config(self) -> bool:
        """Validate the loaded configuration"""
        if not self.metrics_config:
            self.logger.error("No metrics configuration loaded")
            return False
        
        required_fields = ['name', 'expr', 'category']
        
        for category, metrics in self.metrics_config.items():
            for metric in metrics:
                for field in required_fields:
                    if field not in metric:
                        self.logger.error(f"Missing required field '{field}' in metric: {metric.get('name', 'unknown')}")
                        return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def debug_config(self) -> Dict[str, Any]:
        """Get debug information about the configuration"""
        return {
            'config_path': self.config_path,
            'config_exists': os.path.exists(self.config_path),
            'total_metrics': len(self.raw_metrics),
            'categories': list(self.metrics_config.keys()),
            'metrics_by_category': {
                cat: len(metrics) for cat, metrics in self.metrics_config.items()
            },
            'all_metric_names': self.get_all_metric_names(),
            'network_io_metrics': self.get_network_io_metrics()
        }


# Global configuration instance
_config_instance = None

def get_config(config_path: Optional[str] = None) -> ETCDConfig:
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ETCDConfig(config_path)
    return _config_instance

def reset_config():
    """Reset global configuration instance (useful for testing)"""
    global _config_instance
    _config_instance = None