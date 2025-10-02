#!/usr/bin/env python3
"""
OVNK Analyzer MCP Agent Report - Optimized Version
Separated script-based and AI-based analysis with streaming output
"""

import asyncio
import logging
import json
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

class OVNKAnalyzerMCPAgent:
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
        logger.info("Initializing etcd performance analysis session...")
        
        test_id = self.utility.generate_test_id()
        start_time = state.get("start_time")
        end_time = state.get("end_time")
        duration = state.get("duration", "1h")
        
        if start_time and end_time:
            if end_time <= start_time:
                error_msg = "end_time must be after start_time"
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
        state["messages"].append(AIMessage(content=f"Analysis started - Test ID: {test_id}, {mode_info}"))
        
        return state

    async def _collect_metrics_node(self, state: AgentState) -> AgentState:
        """Collect metrics from MCP server"""
        logger.info("Collecting etcd performance metrics via MCP server...")
        
        try:
            params = {"duration": state["duration"]}
            
            if state.get("start_time") and state.get("end_time"):
                params["start_time"] = state["start_time"].isoformat()
                params["end_time"] = state["end_time"].isoformat()
            
            metrics_data = await self._call_mcp_tool("get_etcd_performance_deep_drive", params)
            
            if metrics_data and metrics_data.get("status") == "success":
                state["metrics_data"] = metrics_data
                state["messages"].append(AIMessage(content="Metrics collected successfully"))
            else:
                error_msg = f"Failed to collect metrics: {metrics_data.get('error', 'Unknown error')}"
                state["error"] = error_msg
                state["messages"].append(AIMessage(content=error_msg))
                
        except Exception as e:
            error_msg = f"Error collecting metrics: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            
        return state
            
    async def _analyze_performance_node(self, state: AgentState) -> AgentState:
        """Analyze the collected performance metrics"""
        logger.info("Analyzing etcd performance metrics...")
        
        if state.get("error") or not state.get("metrics_data"):
            return state
            
        try:
            analysis_results = self.report_analyzer.analyze_performance_metrics(
                state["metrics_data"], 
                state["test_id"]
            )
            
            state["analysis_results"] = analysis_results
            state["messages"].append(AIMessage(content="Performance analysis completed"))
            
        except Exception as e:
            error_msg = f"Error analyzing performance: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            
        return state

    async def _script_analysis_node(self, state: AgentState) -> AgentState:
        """Perform script-based root cause analysis"""
        logger.info("Performing script-based root cause analysis...")
        
        if state.get("error") or not state.get("analysis_results"):
            return state
            
        try:
            analysis_results = state["analysis_results"]
            metrics_data = state["metrics_data"]
            
            failed_thresholds = self._identify_failed_thresholds(analysis_results)
            
            if failed_thresholds:
                script_analysis = await self.report_analyzer.script_based_root_cause_analysis(
                    failed_thresholds, metrics_data
                )
                
                state["script_analysis"] = script_analysis
                state["messages"].append(
                    AIMessage(content=f"Script analysis completed - {len(failed_thresholds)} threshold failures")
                )
            else:
                state["messages"].append(
                    AIMessage(content="All thresholds within acceptable ranges")
                )
                
        except Exception as e:
            error_msg = f"Error in script analysis: {str(e)}"
            state["messages"].append(AIMessage(content=error_msg))
            
        return state

    async def _ai_analysis_node(self, state: AgentState) -> AgentState:
        """Perform AI-based root cause analysis"""
        logger.info("Performing AI-based root cause analysis...")
        
        if state.get("error") or not state.get("script_analysis"):
            return state
            
        try:
            analysis_results = state["analysis_results"]
            metrics_data = state["metrics_data"]
            script_analysis = state["script_analysis"]
            
            failed_thresholds = self._identify_failed_thresholds(analysis_results)
            
            if failed_thresholds and script_analysis:
                ai_analysis = await self._ai_root_cause_analysis(
                    failed_thresholds, metrics_data, script_analysis
                )
                
                state["ai_analysis"] = ai_analysis
                state["messages"].append(AIMessage(content="AI analysis completed"))
            else:
                state["messages"].append(AIMessage(content="No AI analysis required"))
                
        except Exception as e:
            error_msg = f"Error in AI analysis: {str(e)}"
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

    async def _ai_root_cause_analysis(self, failed_thresholds: List[Dict[str, Any]], 
                                     metrics_data: Dict[str, Any], 
                                     script_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI to perform advanced root cause analysis"""
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
            response = await self.llm.ainvoke(messages)
            
            # Parse JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'raw_response': response.content}
                
        except Exception as e:
            logger.error(f"Error in AI analysis: {e}")
            return {'error': str(e)}

    async def _generate_report_node(self, state: AgentState) -> AgentState:
        """Generate the performance report"""
        logger.info("Generating performance report...")
        
        if state.get("error") or not state.get("analysis_results"):
            return state
            
        try:
            report = self.report_analyzer.generate_performance_report(
                state["analysis_results"],
                state["test_id"],
                state["duration"]
            )
            
            state["performance_report"] = report
            state["messages"].append(AIMessage(content="Report generated"))
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            state["error"] = error_msg
            state["messages"].append(AIMessage(content=error_msg))
            
        return state

    async def _call_mcp_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server tool via streamable HTTP"""
        try:
            url = f"{self.mcp_server_url}/mcp"
            
            async with streamablehttp_client(url) as (read_stream, write_stream, get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    session_id = get_session_id()
                    logger.info(f"Session ID: {session_id}")
                    
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

    async def run_analysis(self, duration: str = "1h", start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        """Run the complete performance analysis workflow with streaming output"""
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
            print(f"\n{'='*100}")
            print("ETCD PERFORMANCE ANALYSIS - STREAMING OUTPUT")
            print(f"{'='*100}\n")
            
            final_state = None
            
            # Stream through graph execution using async iteration
            async for event in self.graph.astream(initial_state):
                for node_name, node_state in event.items():
                    final_state = node_state
                    
                    print(f"\n[{node_name.upper()}]")
                    
                    # Display latest message
                    if node_state.get("messages"):
                        latest_msg = node_state["messages"][-1]
                        if isinstance(latest_msg, AIMessage):
                            print(f"  Status: {latest_msg.content}")
                    
                    # Display script analysis results
                    if node_name == "script_analysis" and node_state.get("script_analysis"):
                        print(f"\n{'='*80}")
                        print("SCRIPT-BASED ROOT CAUSE ANALYSIS")
                        print(f"{'='*80}")
                        self._print_script_analysis(node_state["script_analysis"])
                    
                    # Display AI analysis results
                    if node_name == "ai_analysis" and node_state.get("ai_analysis"):
                        print(f"\n{'='*80}")
                        print("AI-POWERED ROOT CAUSE ANALYSIS")
                        print(f"{'='*80}")
                        self._print_ai_analysis(node_state["ai_analysis"])
                    
                    # Display final report
                    if node_name == "generate_report" and node_state.get("performance_report"):
                        print(f"\n{'='*80}")
                        print("PERFORMANCE REPORT")
                        print(f"{'='*80}")
                        print(node_state["performance_report"])
            
            # Use final_state from the stream
            if final_state is None:
                final_state = initial_state
            
            return {
                "success": not bool(final_state.get("error")),
                "test_id": final_state.get("test_id"),
                "error": final_state.get("error")
            }
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return {"success": False, "error": str(e)}

    def _print_script_analysis(self, script_analysis: Dict[str, Any]):
        """Print script-based analysis results"""
        disk_analysis = script_analysis.get('disk_io_analysis', {})
        if disk_analysis:
            print("\nDISK I/O ANALYSIS:")
            perf = disk_analysis.get('disk_performance_assessment', {})
            if perf:
                write_perf = perf.get('write_throughput', {})
                print(f"  Write Performance: {write_perf.get('cluster_avg_mb_s', 0)} MB/s")
                print(f"  Grade: {write_perf.get('performance_grade', 'unknown').upper()}")
        
        network_analysis = script_analysis.get('network_analysis', {})
        if network_analysis:
            print("\nNETWORK ANALYSIS:")
            health = network_analysis.get('network_health_assessment', {})
            print(f"  Avg Peer Latency: {health.get('avg_peer_latency_ms', 0)} ms")
            print(f"  Network Grade: {health.get('network_grade', 'unknown').upper()}")

    def _print_ai_analysis(self, ai_analysis: Dict[str, Any]):
        """Print AI-based analysis results"""
        if ai_analysis.get('error'):
            print(f"AI Analysis Error: {ai_analysis['error']}")
            return
        
        if 'raw_response' in ai_analysis:
            print(ai_analysis['raw_response'])
            return
        
        primary = ai_analysis.get('primary_root_cause', {})
        if primary:
            print(f"\nPRIMARY ROOT CAUSE (Confidence: {primary.get('confidence_level', 0)}/10):")
            print(f"  {primary.get('cause', 'Not identified')}")
        
        factors = ai_analysis.get('secondary_factors', [])
        if factors:
            print("\nCONTRIBUTING FACTORS:")
            for i, factor in enumerate(factors, 1):
                print(f"  {i}. {factor}")
        
        recs = ai_analysis.get('recommendations', [])
        if recs:
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(recs, 1):
                priority = rec.get('priority', 'medium').upper()
                print(f"  {i}. [{priority}] {rec.get('action', 'No action')}")

async def main():
    """Main function"""
    print("OVNK etcd Performance Analyzer")
    print("=" * 50)
    
    try:
        agent = OVNKAnalyzerMCPAgent()
        
        mode = input("Select mode (1=Duration, 2=Time Range, default=1): ").strip() or "1"
        
        if mode == "2":
            start_str = input("Start time (YYYY-MM-DD HH:MM:SS UTC): ").strip()
            end_str = input("End time (YYYY-MM-DD HH:MM:SS UTC): ").strip()
            
            if start_str and end_str:
                start_time = pytz.UTC.localize(datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S"))
                end_time = pytz.UTC.localize(datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S"))
                result = await agent.run_analysis(start_time=start_time, end_time=end_time)
            else:
                duration = input("Duration (default: 1h): ").strip() or "1h"
                result = await agent.run_analysis(duration)
        else:
            duration = input("Duration (default: 1h): ").strip() or "1h"
            result = await agent.run_analysis(duration)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"Success: {result['success']}")
        if result.get('test_id'):
            print(f"Test ID: {result['test_id']}")
        if result.get('error'):
            print(f"Error: {result['error']}")
        print(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")