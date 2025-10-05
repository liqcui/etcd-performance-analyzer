# etcd Performance Analyzer

A comprehensive etcd cluster performance analysis and monitoring toolkit built with MCP (Model Context Protocol), LangGraph, and DuckDB. This system provides deep insights into etcd cluster health, performance bottlenecks, and optimization opportunities through both automated script-based analysis and AI-powered root cause detection.

## Overview

The etcd Performance Analyzer is a multi-layered system designed for OpenShift/Kubernetes environments to:

- **Monitor** critical etcd metrics in real-time
- **Analyze** performance patterns and identify bottlenecks
- **Store** historical metrics in DuckDB for trend analysis
- **Generate** comprehensive performance reports
- **Provide** AI-powered root cause analysis
- **Interact** via chat interface for natural language queries

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web UI / Chat Interface                   │
│              (etcd_analyzer_mcp_llm.html)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              FastAPI Chat Server                            │
│         (etcd_analyzer_client_chat.py)                      │
│   • LangGraph Agent with RAG                                │
│   • Streaming responses                                      │
│   • System prompt management                                 │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              MCP Server (FastMCP)                           │
│         (etcd_analyzer_mcp_server.py)                       │
│   • Exposes 11 MCP tools                                    │
│   • Metrics collection                                       │
│   • Performance analysis                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
┌────────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│  Collectors   │ │  Analysis  │ │  Storage   │
│   (tools/)    │ │ (analysis/)│ │ (storage/) │
└───────────────┘ └────────────┘ └────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
         ┌───────────────▼───────────────┐
         │  OpenShift/Kubernetes Cluster │
         │  • Prometheus metrics         │
         │  • etcd pods                  │
         │  • Node resources             │
         └───────────────────────────────┘
```

## Key Components

### 1. MCP Server (`etcd_analyzer_mcp_server.py`)

The core FastMCP server exposing 11 specialized tools:

- **`get_server_health`** - Server health status
- **`get_ocp_cluster_info`** - Cluster infrastructure details
- **`get_etcd_cluster_status`** - etcd member status and leadership
- **`get_etcd_general_info`** - CPU, memory, proposals, leadership metrics
- **`get_etcd_disk_wal_fsync`** - WAL fsync performance (P99 latency)
- **`get_etcd_disk_backend_commit`** - Backend commit latency
- **`get_etcd_disk_io`** - Disk throughput and IOPS
- **`get_etcd_network_io`** - Network latency and utilization
- **`get_etcd_disk_compact_defrag`** - Compaction and defragmentation metrics
- **`get_etcd_performance_deep_drive`** - Comprehensive multi-subsystem analysis
- **`generate_etcd_performance_report`** - Executive-ready performance reports

### 2. Chat Interface (`etcd_analyzer_client_chat.py`)

FastAPI server with LangGraph agent integration:

- **Streaming responses** with SSE (Server-Sent Events)
- **Configurable system prompts** per conversation
- **Tool result formatting** with HTML table support
- **Conversation memory** with thread-based isolation

### 3. Analysis Engines

#### Deep Drive Analyzer (`analysis/etcd_analyzer_performance_deepdrive.py`)
- Collects metrics across all subsystems
- Performs latency pattern analysis
- Identifies bottlenecks (disk, network, memory, consensus)
- Generates root cause analysis with evidence

#### Report Analyzer (`analysis/etcd_analyzer_performance_report.py`)
- Threshold-based analysis (WAL fsync <10ms, backend commit <25ms)
- Baseline comparison against best practices
- Executive summary generation
- Prioritized recommendations

### 4. Agent Workflows

#### Report Agent (`etcd_analyzer_mcp_agent_report.py`)
LangGraph StateGraph workflow:
```
Initialize → Collect Metrics → Analyze Performance 
    → Script Analysis → AI Analysis → Generate Report
```

#### Storage Agent (`etcd_analyzer_mcp_agent_stor2db.py`)
Stores metrics in DuckDB with comprehensive ELT pipeline:
```
Initialize → Cluster Info → General Info → WAL Fsync 
    → Disk I/O → Network I/O → Backend Commit 
    → Compact/Defrag → Store Data → Finalize
```

## Installation

### Prerequisites

```bash
# Python 3.11+
python --version

# OpenShift CLI (oc)
oc version

# Node.js (optional, for MCP Inspector)
node --version
```

### Setup

```bash
# Clone repository
git clone <repository-url>
cd etcd-performance-analyzer

# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your settings:
# - OPENAI_API_KEY (for LLM integration)
# - BASE_URL (LLM endpoint)
```

### Configuration

Edit `config/metrics-etcd.yml` to customize metrics collection:

```yaml
metrics:
  - name: disk_wal_fsync_seconds_duration_p99
    query: 'histogram_quantile(0.99, rate(etcd_disk_wal_fsync_duration_seconds_bucket[5m]))'
    unit: seconds
    description: "99th percentile WAL fsync duration"
```

## Usage

### 1. Start MCP Server

```bash
# Default: http://0.0.0.0:8000
python etcd_analyzer_mcp_server.py

# With MCP Inspector (debugging)
ENABLE_MCP_INSPECTOR=1 python etcd_analyzer_mcp_server.py
```

### 2. Start Chat Interface

```bash
# Default: http://0.0.0.0:8080
python etcd_analyzer_client_chat.py

# Access web UI
open http://localhost:8080/ui
```

### 3. CLI Usage

#### Generate Performance Report

```bash
python etcd_analyzer_mcp_agent_report.py

# Options:
# 1. Duration mode: "1h", "30m", "6h", "1d"
# 2. Time range mode: Start/end timestamps
```

#### Store Metrics to DuckDB

```bash
python etcd_analyzer_mcp_agent_stor2db.py

# Workflow automatically:
# 1. Runs etcd load test (etcd-loads/etcd-load-tools.sh)
# 2. Collects metrics during test
# 3. Stores in etcd_analyzer.duckdb
# 4. Generates summary tables
```

### 4. API Examples

#### Direct Tool Call

```bash
curl -X POST http://localhost:8000/api/tools/get_etcd_disk_wal_fsync \
  -H "Content-Type: application/json" \
  -d '{"duration": "1h"}'
```

#### Chat Completion

```bash
curl -X POST http://localhost:8080/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Analyze etcd WAL fsync performance for the last hour",
    "conversation_id": "session-001"
  }'
```

## Performance Thresholds

The analyzer uses industry best practices:

| Metric | Warning | Critical | Rationale |
|--------|---------|----------|-----------|
| WAL fsync P99 | 5ms | 10ms | Disk write durability |
| Backend commit P99 | 15ms | 25ms | Database performance |
| CPU usage | 70% | 85% | Resource availability |
| Memory usage | 70% | 85% | Prevent OOM |
| Peer latency | 50ms | 100ms | Consensus stability |
| Network utilization | 70% | 85% | Bandwidth capacity |

## Output Examples

### Performance Report

```
==================================================================================================
ETCD PERFORMANCE ANALYSIS REPORT
==================================================================================================
Test ID: perf-20241003-142530
Analysis Duration: 1h
Report Generated: 2024-10-03 14:25:30 UTC
Analysis Status: SUCCESS

EXECUTIVE SUMMARY
==================================================
Overall Disk Performance Health: WARNING
Performance Grade: FAIR

CRITICAL ALERTS
==================================================
[WARNING] WAL fsync latency elevated above 5ms
Impact: Increased write operation latency affecting cluster performance
Action Required: Investigate storage I/O performance

WAL Fsync Performance:
  Health Status: WARNING
  Average Latency: 7.234 ms
  Maximum Latency: 12.456 ms
  Threshold: 10.0 ms
  Pods with Issues: 2/3
```

### DuckDB Storage

Tables created for historical analysis:

- `cluster_info` - Cluster infrastructure
- `general_info_metrics` - CPU, memory, proposals
- `wal_fsync_p99_metrics` - WAL fsync latencies
- `disk_io_throughput` - Disk read/write rates
- `network_io_pod_metrics` - Network performance
- `backend_commit_metrics` - Commit latencies
- `compact_defrag_metrics` - Maintenance operations

## Project Structure

```
etcd-performance-analyzer/
├── analysis/                 # Analysis engines
│   ├── etcd_analyzer_performance_deepdrive.py
│   ├── etcd_analyzer_performance_report.py
│   └── etcd_analyzer_performance_utility.py
├── config/                   # Configuration
│   ├── etcd_config.py
│   └── metrics-etcd.yml
├── elt/                      # ELT transformations
│   └── etcd_analyzer_elt_*.py
├── ocauth/                   # OpenShift authentication
│   └── ocp_auth.py
├── storage/                  # DuckDB storage layer
│   └── etcd_analyzer_stor_*.py
├── tools/                    # Metrics collectors
│   ├── etcd_cluster_status.py
│   ├── etcd_disk_*.py
│   ├── etcd_general_info.py
│   ├── etcd_network_io.py
│   └── ocp_*.py
├── webroot/                  # Web UI
│   └── etcd_analyzer_mcp_llm.html
├── etcd_analyzer_mcp_server.py      # MCP server
├── etcd_analyzer_client_chat.py     # Chat interface
├── etcd_analyzer_mcp_agent_report.py # Report agent
└── etcd_analyzer_mcp_agent_stor2db.py # Storage agent
```

## Troubleshooting

### Authentication Issues

```bash
# Verify OpenShift login
oc whoami
oc get pods -n openshift-etcd

# Check Prometheus access
oc get route -n openshift-monitoring
```

### MCP Server Not Starting

```bash
# Check port availability
lsof -i :8000

# Verify dependencies
pip list | grep -E "fastmcp|pydantic|uvicorn"
```

### No Metrics Returned

```bash
# Verify etcd pods are running
oc get pods -n openshift-etcd

# Check Prometheus queries
oc exec -n openshift-etcd etcd-<pod> -- etcdctl endpoint health
```

## Contributing

Contributions welcome! Key areas:

1. Additional metrics collectors
2. Enhanced analysis algorithms
3. UI/UX improvements
4. Documentation

## License

[Specify your license]

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs

---

**Note**: This tool requires administrative access to OpenShift clusters and is designed for performance analysis and troubleshooting. Always review security implications before deployment in production environments.