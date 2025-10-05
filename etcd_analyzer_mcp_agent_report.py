#!/usr/bin/env python3
"""
OVNK Analyzer MCP Agent Report - Full Streaming Version
All outputs stream in real-time for better user experience
"""

import asyncio
import logging
import json
import sys
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
import pytz
from dataclasses import dataclass
import traceback
import os
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

# MCP imports
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.session import ClientSession

# Import analysis modules
from analysis.etcd_analyzer_performance_report import etcdReportAnalyzer
from analysis.etcd_analyzer_performance_utility import etcdAnalyzerUtility

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def stream_print(text: str, end: str = '\n', flush: bool = True):
    """Print with immediate flush for streaming output"""
    print(text, end=end, flush=flush)
    sys.stdout.flush()

@dataclass
class MCPServerConfig:
    """MCP Server configuration"""
    host: str = "localhost"
    port: int = 8000
    base_url: str = None
    
    def __post_init__(self):
        if self.base_url is None:
            self.base_url = f"http://{self.host}:{self.port}"

class AgentState(TypedDict):
    """State for the LangGraph agent"""
    messages: Annotated[list, add_messages]
    metrics_data: Optional[Dict[str, Any]]
    analysis_results: Optional[Dict[str, Any]]
    script_analysis: Optional[Dict[str, Any]]
    ai_analysis: Optional[Dict[str, Any]]
    performance_report: Optional[str]
    error: Optional[str]
    test_id: Optional[str]
    duration: str
    start_time: Optional[datetime]
    end_time: Optional[datetime]

class etcdAnalyzerMCPAgent:
    """AI agent for etcd performance analysis using MCP server integration"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.utility = etcdAnalyzerUtility()
        self.report_analyzer = etcdReportAnalyzer()
        self.timezone = pytz.UTC
        self.mcp_server_url = mcp_server_url
        
        load_dotenv()
        self.llm = ChatOpenAI(
            model="gemini-2.5-pro",
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1,
            streaming=True
        )
        
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph for the agent workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("initialize", self._initialize_node)
        workflow.add_node("collect_metrics", self._collect_metrics_node)
        workflow.add_node("analyze_performance", self._analyze_performance_node)
        workflow.add_node("script_analysis", self._script_analysis_node)
        workflow.add_node("ai_analysis", self._ai_analysis_node)
        workflow.add_node("generate_report", self._generate_report_node)
        
        # Add edges
        workflow.set_entry_point("initialize")
        workflow.add_edge("initialize", "collect_metrics")
        workflow.add_edge("collect_metrics", "analyze_performance")
        workflow.add_edge("analyze_performance", "script_analysis")
        workflow.add_edge("script_analysis", "ai_analysis")
        workflow.add_edge("ai_analysis", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()

    async def _initialize_node(self, state: AgentState) -> AgentState:
        """Initialize the analysis session"""
        stream_print("\n" + "="*100)
        stream_print("🚀 INITIALIZING ETCD PERFORMANCE ANALYSIS SESSION")
        stream_print("="*100)
        
        test_id = self.utility.generate_test_id()
        start_time = state.get("start_time")
        end_time = state.get("end_time")
        duration = state.get("duration", "1h")
        
        if start_time and end_time:
            if end_time <= start_time:
                error_msg = "❌ Error: end_time must be after start_time"
                stream_print(error_msg)
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                return state
            
            calculated_duration = end_time - start_time
            hours = int(calculated_duration.total_seconds() / 3600)
            minutes = int((calculated_duration.total_seconds() % 3600) / 60)
            duration_display = f"{hours}h{minutes}m" if hours else f"{minutes}m"
            
            mode_info = f"Time Range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')} ({duration_display})"
        else:
            mode_info = f"Duration: {duration}"
        
        state["test_id"] = test_id
        state["duration"] = duration
        
        stream_print(f"📋 Test ID: {test_id}")
        stream_print(f"⏱️  {mode_info}")
        stream_print(f"🔗 MCP Server: {self.mcp_server_url}")
        
        state["messages"].append(AIMessage(content=f"Analysis started - Test ID: {test_id}, {mode_info}"))
        
        return state

    async def _collect_metrics_node(self, state: AgentState) -> AgentState:
        """Collect metrics from MCP server"""
        stream_print("\n" + "="*100)
        stream_print("📊 COLLECTING ETCD PERFORMANCE METRICS")
        stream_print("="*100)
        
        try:
            params = {"duration": state["duration"]}
            
            if state.get("start_time") and state.get("end_time"):
                params["start_time"] = state["start_time"].isoformat()
                params["end_time"] = state["end_time"].isoformat()
            
            stream_print(f"🔍 Querying MCP server with params: {params}")
            stream_print("⏳ Fetching metrics...", end="")
            
            metrics_data = await self._call_mcp_tool("get_etcd_performance_deep_drive", params)
            
            if metrics_data and metrics_data.get("status") == "success":
                state["metrics_data"] = metrics_data
                data = metrics_data.get('data', {})
                
                stream_print(" ✅ Done")
                stream_print("\n📦 Metrics Summary:")
                stream_print(f"  • WAL fsync metrics: {len(data.get('wal_fsync_data', []))} records")
                stream_print(f"  • Backend commit metrics: {len(data.get('backend_commit_data', []))} records")
                stream_print(f"  • General info metrics: {len(data.get('general_info_data', []))} records")
                stream_print(f"  • Disk metrics: {len(data.get('disk_metrics_data', []))} records")
                stream_print(f"  • Network metrics: {len(data.get('network_metrics_data', []))} records")
                
                state["messages"].append(AIMessage(content="Metrics collected successfully"))
            else:
                error_msg = f"❌ Failed to collect metrics: {metrics_data.get('error', 'Unknown error')}"
                stream_print(f" {error_msg}")
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                
        except Exception as e:
            error_msg = f"❌ Error collecting metrics: {str(e)}"
            stream_print(f" {error_msg}")
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            
        return state
            
    async def _analyze_performance_node(self, state: AgentState) -> AgentState:
        """Analyze the collected performance metrics"""
        stream_print("\n" + "="*100)
        stream_print("🔬 ANALYZING PERFORMANCE METRICS")
        stream_print("="*100)
        
        if state.get("error") or not state.get("metrics_data"):
            stream_print("⏭️  Skipping analysis due to previous errors")
            return state
            
        try:
            stream_print("⏳ Processing metrics...", end="")
            
            analysis_results = self.report_analyzer.analyze_performance_metrics(
                state["metrics_data"], 
                state["test_id"]
            )
            
            state["analysis_results"] = analysis_results
            
            stream_print(" ✅ Done")
            
            # Stream key findings
            stream_print("\n🎯 Key Findings:")
            critical = analysis_results.get('critical_metrics_analysis', {})
            
            wal = critical.get('wal_fsync_analysis', {})
            if wal:
                status = wal.get('health_status', 'unknown').upper()
                emoji = "🔴" if status == "CRITICAL" else "🟡" if status == "WARNING" else "🟢"
                stream_print(f"  {emoji} WAL fsync: {status}")
            
            backend = critical.get('backend_commit_analysis', {})
            if backend:
                status = backend.get('health_status', 'unknown').upper()
                emoji = "🔴" if status == "CRITICAL" else "🟡" if status == "WARNING" else "🟢"
                stream_print(f"  {emoji} Backend commit: {status}")
            
            perf_summary = analysis_results.get('performance_summary', {})
            cpu = perf_summary.get('cpu_analysis', {})
            if cpu:
                status = cpu.get('health_status', 'unknown').upper()
                emoji = "🔴" if status == "CRITICAL" else "🟡" if status == "WARNING" else "🟢"
                stream_print(f"  {emoji} CPU utilization: {status}")
            
            state["messages"].append(AIMessage(content="Performance analysis completed"))
            
        except Exception as e:
            error_msg = f"❌ Error analyzing performance: {str(e)}"
            stream_print(error_msg)
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            
        return state

    async def _script_analysis_node(self, state: AgentState) -> AgentState:
        """Perform script-based root cause analysis"""
        stream_print("\n" + "="*100)
        stream_print("🔍 SCRIPT-BASED ROOT CAUSE ANALYSIS")
        stream_print("="*100)
        
        if state.get("error") or not state.get("analysis_results"):
            stream_print("⏭️  Skipping script analysis due to previous errors")
            return state
            
        try:
            analysis_results = state["analysis_results"]
            metrics_data = state["metrics_data"]
            
            stream_print("🔎 Identifying threshold failures...", end="")
            failed_thresholds = self._identify_failed_thresholds(analysis_results)
            stream_print(f" Found {len(failed_thresholds)} failures")
            
            if failed_thresholds:
                stream_print("\n⚠️  Failed Thresholds:")
                for threshold in failed_thresholds:
                    metric = threshold['metric']
                    current = threshold['current_value']
                    threshold_val = threshold['threshold_value']
                    severity = threshold['severity'].upper()
                    emoji = "🔴" if severity == "CRITICAL" else "🟡"
                    stream_print(f"  {emoji} {metric}: {current:.2f} (threshold: {threshold_val})")
                
                stream_print("\n⏳ Performing deep analysis...", end="")
                script_analysis = await self.report_analyzer.script_based_root_cause_analysis(
                    failed_thresholds, metrics_data
                )
                stream_print(" ✅ Done")
                
                state["script_analysis"] = script_analysis
                
                # Stream script analysis results
                self._stream_script_analysis(script_analysis)
                
                state["messages"].append(
                    AIMessage(content=f"Script analysis completed - {len(failed_thresholds)} threshold failures")
                )
            else:
                stream_print("✅ All thresholds within acceptable ranges")
                state["messages"].append(
                    AIMessage(content="All thresholds within acceptable ranges")
                )
                
        except Exception as e:
            error_msg = f"❌ Error in script analysis: {str(e)}"
            stream_print(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            
        return state

    async def _ai_analysis_node(self, state: AgentState) -> AgentState:
        """Perform AI-based root cause analysis"""
        stream_print("\n" + "="*100)
        stream_print("🤖 AI-POWERED ROOT CAUSE ANALYSIS")
        stream_print("="*100)
        
        if state.get("error") or not state.get("script_analysis"):
            stream_print("⏭️  Skipping AI analysis")
            return state
            
        try:
            analysis_results = state["analysis_results"]
            metrics_data = state["metrics_data"]
            script_analysis = state["script_analysis"]
            
            failed_thresholds = self._identify_failed_thresholds(analysis_results)
            
            if failed_thresholds and script_analysis:
                stream_print("🧠 Invoking AI model for deep analysis...")
                stream_print("⏳ Processing with LLM (streaming)...\n")
                
                ai_analysis = await self._ai_root_cause_analysis_streaming(
                    failed_thresholds, metrics_data, script_analysis
                )
                
                state["ai_analysis"] = ai_analysis
                state["messages"].append(AIMessage(content="AI analysis completed"))
            else:
                stream_print("ℹ️  No AI analysis required")
                state["messages"].append(AIMessage(content="No AI analysis required"))
                
        except Exception as e:
            error_msg = f"❌ Error in AI analysis: {str(e)}"
            stream_print(error_msg)
            state["messages"].append(AIMessage(content=error_msg))
            
        return state

    def _identify_failed_thresholds(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify which thresholds have failed"""
        failed_thresholds = []
        
        try:
            critical_analysis = analysis_results.get('critical_metrics_analysis', {})
            performance_summary = analysis_results.get('performance_summary', {})
            
            # WAL fsync failures
            wal_analysis = critical_analysis.get('wal_fsync_analysis', {})
            if wal_analysis.get('health_status') in ['critical', 'warning']:
                cluster_summary = wal_analysis.get('cluster_summary', {})
                failed_thresholds.append({
                    'metric': 'wal_fsync_p99',
                    'threshold_type': 'latency',
                    'current_value': cluster_summary.get('avg_latency_ms', 0),
                    'threshold_value': 10.0,
                    'severity': wal_analysis.get('health_status'),
                    'pods_affected': cluster_summary.get('pods_with_issues', 0),
                    'total_pods': cluster_summary.get('total_pods', 0)
                })
            
            # Backend commit failures
            backend_analysis = critical_analysis.get('backend_commit_analysis', {})
            if backend_analysis.get('health_status') in ['critical', 'warning']:
                cluster_summary = backend_analysis.get('cluster_summary', {})
                failed_thresholds.append({
                    'metric': 'backend_commit_p99',
                    'threshold_type': 'latency',
                    'current_value': cluster_summary.get('avg_latency_ms', 0),
                    'threshold_value': 25.0,
                    'severity': backend_analysis.get('health_status'),
                    'pods_affected': cluster_summary.get('pods_with_issues', 0),
                    'total_pods': cluster_summary.get('total_pods', 0)
                })
            
            # CPU failures
            cpu_analysis = performance_summary.get('cpu_analysis', {})
            if cpu_analysis.get('health_status') in ['critical', 'warning']:
                cluster_summary = cpu_analysis.get('cluster_summary', {})
                failed_thresholds.append({
                    'metric': 'cpu_usage',
                    'threshold_type': 'utilization',
                    'current_value': cluster_summary.get('avg_usage', 0),
                    'threshold_value': 70.0,
                    'severity': cpu_analysis.get('health_status'),
                    'pods_affected': cluster_summary.get('critical_pods', 0) + cluster_summary.get('warning_pods', 0),
                    'total_pods': cluster_summary.get('total_pods', 0)
                })
                
        except Exception as e:
            logger.error(f"Error identifying failed thresholds: {e}")
            
        return failed_thresholds

    async def _ai_root_cause_analysis_streaming(self, failed_thresholds: List[Dict[str, Any]], 
                                                metrics_data: Dict[str, Any], 
                                                script_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to perform advanced root cause analysis with streaming output"""
        try:
            context = self._prepare_cluster_overview(metrics_data)
            
            prompt = f"""You are an expert etcd performance analyst. Analyze the following data and provide root cause analysis.

Failed Thresholds:
{json.dumps(failed_thresholds, indent=2)}

Script-based Analysis:
{json.dumps(script_analysis, indent=2)}

Cluster Overview:
{json.dumps(context, indent=2)}

Provide a JSON response with:
1. primary_root_cause: {{cause, confidence_level (1-10), explanation}}
2. secondary_factors: [list of contributing factors]
3. evidence: [supporting evidence]
4. recommendations: [{{priority, action, expected_impact}}]
5. risk_assessment: overall risk description

Focus on: Disk I/O, CPU, Network, Memory, Database maintenance"""
            
            messages = [HumanMessage(content=prompt)]
            
            # Stream the AI response
            full_response = ""
            async for chunk in self.llm.astream(messages):
                content = chunk.content
                if content:
                    stream_print(content, end="")
                    full_response += content
            
            stream_print("\n")  # Newline after streaming
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', full_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                self._stream_ai_analysis_structured(parsed)
                return parsed
            else:
                return {'raw_response': full_response}
                
        except Exception as e:
            error_msg = f"❌ Error in AI analysis: {str(e)}"
            stream_print(error_msg)
            return {'error': str(e)}

    async def _generate_report_node(self, state: AgentState) -> AgentState:
        """Generate the performance report"""
        stream_print("\n" + "="*100)
        stream_print("📝 GENERATING PERFORMANCE REPORT")
        stream_print("="*100)
        
        if state.get("error") or not state.get("analysis_results"):
            stream_print("⏭️  Skipping report generation due to previous errors")
            return state
            
        try:
            stream_print("⏳ Compiling report...", end="")
            
            report = self.report_analyzer.generate_performance_report(
                state["analysis_results"],
                state["test_id"],
                state["duration"]
            )
            
            stream_print(" ✅ Done\n")
            
            state["performance_report"] = report
            
            # Stream the report
            stream_print("="*100)
            stream_print("📊 COMPLETE PERFORMANCE ANALYSIS REPORT")
            stream_print("="*100)
            stream_print(report)
            stream_print("="*100)
            
            state["messages"].append(AIMessage(content="Performance report generated successfully"))
            logger.info("Performance report generation completed")
            
        except Exception as e:
            error_msg = f"❌ Error generating report: {str(e)}"
            stream_print(error_msg)
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            
        return state

    async def _call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server tool via streamable HTTP"""
        try:
            url = f"{self.mcp_server_url}/mcp"
            
            async with streamablehttp_client(url) as (read_stream, write_stream, get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    session_id = get_session_id()
                    logger.debug(f"MCP Session ID: {session_id}")
                    
                    result = await session.call_tool(tool_name, params or {})
                    
                    if not result or not getattr(result, "content", None):
                        return {"status": "error", "error": "Empty response"}

                    first_item = result.content[0]
                    text_payload = getattr(first_item, "text", None) or (first_item.get("text") if isinstance(first_item, dict) else None)

                    if text_payload is None:
                        return {"status": "error", "error": "Non-text content"}

                    return json.loads(text_payload)
                        
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"status": "error", "error": str(e)}

    def _prepare_cluster_overview(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare cluster overview for AI analysis"""
        overview = {}
        
        try:
            data = metrics_data.get('data', {})
            
            wal_data = data.get('wal_fsync_data', [])
            if wal_data:
                avg_latencies = [m.get('avg', 0) * 1000 for m in wal_data if 'p99' in m.get('metric_name', '')]
                overview['wal_fsync_avg_ms'] = round(sum(avg_latencies) / len(avg_latencies), 3) if avg_latencies else 0
            
            backend_data = data.get('backend_commit_data', [])
            if backend_data:
                avg_latencies = [m.get('avg', 0) * 1000 for m in backend_data if 'p99' in m.get('metric_name', '')]
                overview['backend_commit_avg_ms'] = round(sum(avg_latencies) / len(avg_latencies), 3) if avg_latencies else 0
            
            general_data = data.get('general_info_data', [])
            cpu_metrics = [m for m in general_data if 'cpu_usage' in m.get('metric_name', '')]
            if cpu_metrics:
                avg_cpu = sum(m.get('avg', 0) for m in cpu_metrics) / len(cpu_metrics)
                overview['avg_cpu_usage_percent'] = round(avg_cpu, 2)
                
        except Exception as e:
            logger.error(f"Error preparing overview: {e}")
            
        return overview

    def _stream_script_analysis(self, script_analysis: Dict[str, Any]):
        """Stream script-based analysis results"""
        stream_print("\n📋 Analysis Results:")
        
        disk_analysis = script_analysis.get('disk_io_analysis', {})
        if disk_analysis:
            stream_print("\n  💾 DISK I/O ANALYSIS:")
            perf = disk_analysis.get('disk_performance_assessment', {})
            if perf:
                write_perf = perf.get('write_throughput', {})
                grade = write_perf.get('performance_grade', 'unknown').upper()
                emoji = "🟢" if grade == "EXCELLENT" else "🟡" if grade == "GOOD" else "🔴"
                stream_print(f"     {emoji} Write Performance: {write_perf.get('cluster_avg_mb_s', 0):.2f} MB/s")
                stream_print(f"     Grade: {grade}")
        
        network_analysis = script_analysis.get('network_analysis', {})
        if network_analysis:
            stream_print("\n  🌐 NETWORK ANALYSIS:")
            health = network_analysis.get('network_health_assessment', {})
            grade = health.get('network_grade', 'unknown').upper()
            emoji = "🟢" if grade == "EXCELLENT" else "🟡" if grade == "GOOD" else "🔴"
            stream_print(f"     {emoji} Avg Peer Latency: {health.get('avg_peer_latency_ms', 0):.2f} ms")
            stream_print(f"     Network Grade: {grade}")

    def _stream_ai_analysis_structured(self, ai_analysis: Dict[str, Any]):
        """Stream structured AI analysis results"""
        stream_print("\n\n📋 Structured Analysis:")
        
        primary = ai_analysis.get('primary_root_cause', {})
        if primary:
            confidence = primary.get('confidence_level', 0)
            emoji = "🔴" if confidence >= 8 else "🟡" if confidence >= 5 else "🟢"
            stream_print(f"\n  {emoji} PRIMARY ROOT CAUSE (Confidence: {confidence}/10):")
            stream_print(f"     {primary.get('cause', 'Not identified')}")
            if primary.get('explanation'):
                stream_print(f"     💡 {primary['explanation']}")
        
        factors = ai_analysis.get('secondary_factors', [])
        if factors:
            stream_print("\n  📌 CONTRIBUTING FACTORS:")
            for i, factor in enumerate(factors, 1):
                stream_print(f"     {i}. {factor}")
        
        recs = ai_analysis.get('recommendations', [])
        if recs:
            stream_print("\n  💡 RECOMMENDATIONS:")
            for i, rec in enumerate(recs, 1):
                priority = rec.get('priority', 'medium').upper()
                emoji = "🔴" if priority == "HIGH" else "🟡" if priority == "MEDIUM" else "🟢"
                stream_print(f"     {emoji} [{priority}] {rec.get('action', 'No action')}")
                if rec.get('expected_impact'):
                    stream_print(f"        Impact: {rec['expected_impact']}")

    async def run_analysis(self, duration: str = "1h", start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        """Run the complete performance analysis workflow with full streaming output"""
        logger.info("Starting etcd performance analysis...")
        
        initial_state = {
            "messages": [
                SystemMessage(content="OVNK etcd Performance Analyzer"),
                HumanMessage(content=f"Analyze etcd performance")
            ],
            "metrics_data": None,
            "analysis_results": None,
            "script_analysis": None,
            "ai_analysis": None,
            "performance_report": None,
            "error": None,
            "test_id": None,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time
        }
        
        try:
            final_state = None
            
            # Stream through graph execution using async iteration
            async for event in self.graph.astream(initial_state):
                for node_name, node_state in event.items():
                    final_state = node_state
            
            # Use final_state from the stream
            if final_state is None:
                final_state = initial_state
            
            # Print final summary
            stream_print("\n" + "="*100)
            stream_print("✅ ANALYSIS COMPLETE")
            stream_print("="*100)
            stream_print(f"Success: {'✅ Yes' if not final_state.get('error') else '❌ No'}")
            if final_state.get('test_id'):
                stream_print(f"Test ID: {final_state['test_id']}")
            if final_state.get('error'):
                stream_print(f"Error: {final_state['error']}")
            stream_print("="*100 + "\n")
            
            return {
                "success": not bool(final_state.get("error")),
                "test_id": final_state.get("test_id"),
                "error": final_state.get("error"),
                "analysis_results": final_state.get("analysis_results"),
                "performance_report": final_state.get("performance_report")
            }
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            stream_print(f"\n❌ FATAL ERROR: {str(e)}\n")
            return {"success": False, "error": str(e)}

async def main():
    """Main function"""
    stream_print("\n" + "="*100)
    stream_print("🔧 OVNK ETCD PERFORMANCE ANALYZER")
    stream_print("="*100)
    
    try:
        agent = etcdAnalyzerMCPAgent()
        
        mode = input("\n📝 Select mode (1=Duration, 2=Time Range, default=1): ").strip() or "1"
        
        if mode == "2":
            start_str = input("📅 Start time (YYYY-MM-DD HH:MM:SS UTC): ").strip()
            end_str = input("📅 End time (YYYY-MM-DD HH:MM:SS UTC): ").strip()
            
            if start_str and end_str:
                start_time = pytz.UTC.localize(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S"))
                end_time = pytz.UTC.localize(datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S"))
                result = await agent.run_analysis(start_time=start_time, end_time=end_time)
            else:
                duration = input("⏱️  Duration (default: 1h): ").strip() or "1h"
                result = await agent.run_analysis(duration)
        else:
            duration = input("⏱️  Duration (default: 1h): ").strip() or "1h"
            result = await agent.run_analysis(duration)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        stream_print(f"\n❌ ERROR: {str(e)}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        stream_print("\n\n⚠️  Analysis interrupted by user\n")