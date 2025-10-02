"""
OpenShift Authentication Module
Handles authentication and service discovery for OpenShift clusters
"""

import os
import base64
import logging
import warnings
from typing import Optional, Dict, Any, Tuple
import shlex
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import asyncio
import subprocess

# Suppress urllib3 SSL warnings for self-signed certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class OCPAuth:
    """OpenShift authentication and service discovery"""
    
    def __init__(self):
        self.k8s_client = None
        self.prometheus_url = None
        self.token = None
        self.ca_cert_path = None
        self.logger = logging.getLogger(__name__)
        # Store full discovery details for building fallbacks
        self.prometheus_info: Optional[Dict[str, Any]] = None
        
    async def initialize(self) -> bool:
        """Initialize Kubernetes client and discover services"""
        try:
            # Load kubeconfig
            config.load_kube_config()
            self.k8s_client = client.ApiClient()
            
            # Get authentication details
            await self._get_auth_details()
            
            # Discover Prometheus service
            prometheus_info = await self._discover_prometheus()
            if prometheus_info:
                self.prometheus_info = prometheus_info
                self.prometheus_url = prometheus_info['url']
                self.logger.info(f"Discovered Prometheus at: {self.prometheus_url}")
                return True
            else:
                self.logger.error("Failed to discover Prometheus service")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize OCP authentication: {e}")
            return False
    
    async def _get_auth_details(self):
        """Extract authentication details from kubeconfig"""
        # Primary fallback: try to create service account token for Prometheus
        if not self.token:
            self.token = await self._create_prometheus_sa_token()
        
        # Secondary fallback: try 'oc whoami -t' if SA token creation failed
        if not self.token:
            try:
                self.logger.info("Attempting to get token via 'oc whoami -t'")
                proc = await asyncio.create_subprocess_exec(
                    'oc', 'whoami', '-t',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode == 0:
                    token = stdout.decode().strip()
                    if token and len(token) > 10:  # Basic sanity check
                        self.token = token
                        self.logger.info("Successfully obtained token via 'oc whoami -t'")
                    else:
                        self.logger.warning("Token from 'oc whoami -t' appears invalid")
                else:
                    err = stderr.decode().strip()
                    if err:
                        self.logger.debug(f"'oc whoami -t' failed: {err}")
            except Exception as e:
                self.logger.debug(f"Failed to run 'oc whoami -t' for token fallback: {e}")
        
        # Final fallback: try to get token from service account
        if not self.token:
            try:
                sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"
                if os.path.exists(sa_token_path):
                    with open(sa_token_path, 'r') as f:
                        self.token = f.read().strip()
                    self.logger.info("Using service account token")
            except Exception as e:
                self.logger.debug(f"Could not read service account token: {e}")
        
        if not self.token:
            self.logger.warning("No authentication token found - Prometheus access may fail")
        else:
            # Don't log the full token, just confirm we have one
            self.logger.info(f"Authentication token configured (length: {len(self.token)})")
    
    async def _create_prometheus_sa_token(self) -> Optional[str]:
        """Create service account token for Prometheus access"""
        # List of service accounts to try in order of preference
        sa_configs = [
            {'namespace': 'openshift-monitoring', 'name': 'prometheus-k8s'},
            {'namespace': 'openshift-user-workload-monitoring', 'name': 'prometheus-k8s'},
            {'namespace': 'openshift-monitoring', 'name': 'prometheus'},
            {'namespace': 'monitoring', 'name': 'prometheus'},
        ]
        
        for sa_config in sa_configs:
            namespace = sa_config['namespace']
            sa_name = sa_config['name']
            
            self.logger.info(f"Attempting to create token for SA {sa_name} in namespace {namespace}")
            
            # Method 1: Try 'oc create token' (newer method, preferred)
            token = await self._try_oc_create_token(namespace, sa_name)
            if token:
                self.logger.info(f"Successfully created token using 'oc create token' for {namespace}/{sa_name}")
                return token
            
            # Method 2: Try 'oc sa new-token' (older method, fallback)
            token = await self._try_oc_sa_new_token(namespace, sa_name)
            if token:
                self.logger.info(f"Successfully created token using 'oc sa new-token' for {namespace}/{sa_name}")
                return token
            
            self.logger.debug(f"Failed to create token for {namespace}/{sa_name}")
        
        self.logger.warning("Failed to create service account token for any Prometheus service account")
        return None
    
    async def _try_oc_create_token(self, namespace: str, sa_name: str) -> Optional[str]:
        """Try to create token using 'oc create token' command"""
        try:
            # Set KUBECONFIG environment if it exists
            env = os.environ.copy()
            
            cmd = ['oc', 'create', 'token', sa_name, '-n', namespace]
            
            # Add duration for token validity (24 hours)
            cmd.extend(['--duration', '24h'])
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                token = stdout.decode().strip()
                if token and len(token) > 20:  # Basic sanity check for JWT tokens
                    return token
                else:
                    self.logger.debug(f"'oc create token' returned invalid token for {namespace}/{sa_name}")
            else:
                stderr_text = stderr.decode().strip()
                if stderr_text:
                    self.logger.debug(f"'oc create token' failed for {namespace}/{sa_name}: {stderr_text}")
                
        except Exception as e:
            self.logger.debug(f"Error running 'oc create token' for {namespace}/{sa_name}: {e}")
        
        return None
    
    async def _try_oc_sa_new_token(self, namespace: str, sa_name: str) -> Optional[str]:
        """Try to create token using 'oc sa new-token' command (legacy)"""
        try:
            # Set KUBECONFIG environment if it exists
            env = os.environ.copy()
            
            cmd = ['oc', 'sa', 'new-token', sa_name, '-n', namespace]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                token = stdout.decode().strip()
                if token and len(token) > 20:  # Basic sanity check
                    return token
                else:
                    self.logger.debug(f"'oc sa new-token' returned invalid token for {namespace}/{sa_name}")
            else:
                stderr_text = stderr.decode().strip()
                if stderr_text:
                    self.logger.debug(f"'oc sa new-token' failed for {namespace}/{sa_name}: {stderr_text}")
                
        except Exception as e:
            self.logger.debug(f"Error running 'oc sa new-token' for {namespace}/{sa_name}: {e}")
        
        return None
    
    async def _verify_sa_exists(self, namespace: str, sa_name: str) -> bool:
        """Verify that service account exists"""
        try:
            if not self.k8s_client:
                return False
                
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Check if service account exists
            try:
                v1.read_namespaced_service_account(name=sa_name, namespace=namespace)
                return True
            except ApiException as e:
                if e.status == 404:
                    self.logger.debug(f"Service account {namespace}/{sa_name} not found")
                else:
                    self.logger.debug(f"Error checking service account {namespace}/{sa_name}: {e}")
                return False
                
        except Exception as e:
            self.logger.debug(f"Error verifying service account {namespace}/{sa_name}: {e}")
            return False
    
    async def _create_sa_if_missing(self, namespace: str, sa_name: str) -> bool:
        """Create service account if it doesn't exist (for testing purposes)"""
        try:
            if not self.k8s_client:
                return False
                
            # First check if it exists
            if await self._verify_sa_exists(namespace, sa_name):
                return True
            
            self.logger.debug(f"Service account {namespace}/{sa_name} not found, attempting to create")
            
            # Try to create using oc command (safer than direct API calls for permissions)
            env = os.environ.copy()
            cmd = ['oc', 'create', 'sa', sa_name, '-n', namespace]
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            stdout, stderr = await proc.communicate()
            
            if proc.returncode == 0:
                self.logger.info(f"Created service account {namespace}/{sa_name}")
                return True
            else:
                stderr_text = stderr.decode().strip()
                self.logger.debug(f"Failed to create service account {namespace}/{sa_name}: {stderr_text}")
                return False
                
        except Exception as e:
            self.logger.debug(f"Error creating service account {namespace}/{sa_name}: {e}")
            return False
    
    async def _discover_prometheus(self) -> Optional[Dict[str, Any]]:
        """Discover Prometheus service in OpenShift monitoring namespace - routes first, then services"""
        try:
            # Common monitoring namespaces to check
            monitoring_namespaces = [
                'openshift-monitoring',
                'openshift-user-workload-monitoring',
                'monitoring',
                'kube-system'
            ]
            
            # First, try to find Prometheus through OpenShift routes (preferred)
            for namespace in monitoring_namespaces:
                try:
                    route_info = await self._find_prometheus_route(namespace)
                    if route_info:
                        self.logger.info(f"Found Prometheus route in namespace: {namespace}")
                        return {
                            'url': route_info['url'],
                            'namespace': namespace,
                            'access_method': 'route',
                            'route_host': route_info['host'],
                            'tls_enabled': route_info['tls_enabled']
                        }
                except Exception as e:
                    self.logger.debug(f"No route found in namespace {namespace}: {e}")
                    continue
            
            # If no routes found, fall back to service discovery
            self.logger.info("No Prometheus routes found, trying service discovery...")
            return await self._discover_prometheus_service()
            
        except Exception as e:
            self.logger.error(f"Error discovering Prometheus: {e}")
            return None
    
    async def _find_prometheus_route(self, namespace: str) -> Optional[Dict[str, Any]]:
        """Find OpenShift route for Prometheus"""
        try:
            # Use dynamic client for routes (OpenShift specific)
            from kubernetes import dynamic
            from kubernetes.client import api_client
            
            dyn_client = dynamic.DynamicClient(api_client.ApiClient())
            route_api = dyn_client.resources.get(api_version='route.openshift.io/v1', kind='Route')
            
            # Try different label selectors for Prometheus routes
            label_selectors = [
                "app.kubernetes.io/name=prometheus",
                "app=prometheus",
                "component=prometheus"
            ]
            
            for label_selector in label_selectors:
                try:
                    routes = route_api.get(namespace=namespace, label_selector=label_selector)
                    
                    if routes.items:
                        route = routes.items[0]
                        host = route.spec.host
                        tls = route.spec.get('tls')
                        tls_enabled = bool(tls)
                        scheme = 'https' if tls_enabled else 'http'
                        
                        return {
                            'url': f"{scheme}://{host}",
                            'host': host,
                            'tls_enabled': tls_enabled,
                            'route_name': route.metadata.name
                        }
                except Exception as e:
                    self.logger.debug(f"No routes found with selector '{label_selector}' in {namespace}: {e}")
                    continue
            
            # Also try to find routes by name patterns
            try:
                all_routes = route_api.get(namespace=namespace)
                for route in all_routes.items:
                    route_name = route.metadata.name.lower()
                    if any(name in route_name for name in ['prometheus', 'monitoring']):
                        host = route.spec.host
                        tls = route.spec.get('tls')
                        tls_enabled = bool(tls)
                        scheme = 'https' if tls_enabled else 'http'
                        
                        return {
                            'url': f"{scheme}://{host}",
                            'host': host,
                            'tls_enabled': tls_enabled,
                            'route_name': route.metadata.name
                        }
            except Exception as e:
                self.logger.debug(f"Error checking route names in {namespace}: {e}")
                
        except Exception as e:
            self.logger.debug(f"Could not find route in namespace {namespace}: {e}")
        
        return None
    
    async def _discover_prometheus_service(self) -> Optional[Dict[str, Any]]:
        """Discover Prometheus through service discovery"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Common monitoring namespaces to check
            monitoring_namespaces = [
                'openshift-monitoring',
                'openshift-user-workload-monitoring',
                'monitoring',
                'kube-system'
            ]
            
            for namespace in monitoring_namespaces:
                try:
                    # Look for Prometheus service
                    services = v1.list_namespaced_service(
                        namespace=namespace,
                        label_selector="app.kubernetes.io/name=prometheus"
                    )
                    
                    if not services.items:
                        # Try alternative labels
                        services = v1.list_namespaced_service(
                            namespace=namespace,
                            label_selector="app=prometheus"
                        )
                    
                    if services.items:
                        service = services.items[0]
                        service_name = service.metadata.name
                        
                        # Get service details
                        port = None
                        for svc_port in service.spec.ports:
                            if svc_port.name in ['web', 'http', 'prometheus'] or svc_port.port in [9090, 9091]:
                                port = svc_port.port
                                break
                        
                        if not port and service.spec.ports:
                            port = service.spec.ports[0].port
                        
                        # Build Prometheus URL based on service type
                        prometheus_url = None
                        access_method = None
                        
                        if service.spec.type == 'LoadBalancer' and service.status.load_balancer.ingress:
                            host = service.status.load_balancer.ingress[0].ip or service.status.load_balancer.ingress[0].hostname
                            prometheus_url = f"http://{host}:{port}"
                            access_method = 'loadbalancer'
                        elif service.spec.type == 'NodePort':
                            # Get any node IP
                            nodes = v1.list_node()
                            if nodes.items:
                                node_ip = None
                                for address in nodes.items[0].status.addresses:
                                    if address.type in ['ExternalIP', 'InternalIP']:
                                        node_ip = address.address
                                        break
                                if node_ip:
                                    prometheus_url = f"http://{node_ip}:{service.spec.ports[0].node_port}"
                                    access_method = 'nodeport'
                        
                        if not prometheus_url:
                            # Use cluster internal URL as fallback
                            prometheus_url = f"http://{service_name}.{namespace}.svc.cluster.local:{port}"
                            access_method = 'cluster_internal'
                        
                        self.logger.info(f"Found Prometheus service in namespace: {namespace}")
                        return {
                            'url': prometheus_url,
                            'namespace': namespace,
                            'service_name': service_name,
                            'port': port,
                            'access_method': access_method
                        }
                        
                except ApiException as e:
                    if e.status != 404:  # Ignore namespace not found errors
                        self.logger.warning(f"Error checking namespace {namespace}: {e}")
                    continue
            
            # If no service found, try to find Prometheus pods directly
            self.logger.info("No Prometheus services found, trying pod discovery...")
            return await self._discover_prometheus_pods()
            
        except Exception as e:
            self.logger.error(f"Error discovering Prometheus services: {e}")
            return None
    
    async def _discover_prometheus_pods(self) -> Optional[Dict[str, Any]]:
        """Discover Prometheus through pods if service discovery fails"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            monitoring_namespaces = [
                'openshift-monitoring',
                'openshift-user-workload-monitoring', 
                'monitoring'
            ]
            
            for namespace in monitoring_namespaces:
                try:
                    pods = v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector="app.kubernetes.io/name=prometheus"
                    )
                    
                    if not pods.items:
                        # Try alternative labels
                        pods = v1.list_namespaced_pod(
                            namespace=namespace,
                            label_selector="app=prometheus"
                        )
                    
                    if pods.items:
                        pod = pods.items[0]
                        pod_ip = pod.status.pod_ip
                        
                        # Default Prometheus port
                        port = 9090
                        
                        # Try to get port from container
                        for container in pod.spec.containers:
                            if container.ports:
                                for container_port in container.ports:
                                    if container_port.name in ['web', 'http', 'prometheus']:
                                        port = container_port.container_port
                                        break
                        
                        prometheus_url = f"http://{pod_ip}:{port}"
                        
                        self.logger.info(f"Found Prometheus pod in namespace: {namespace}")
                        return {
                            'url': prometheus_url,
                            'namespace': namespace,
                            'pod_name': pod.metadata.name,
                            'port': port,
                            'access_method': 'direct_pod'
                        }
                        
                except ApiException as e:
                    if e.status != 404:
                        self.logger.warning(f"Error checking pods in namespace {namespace}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error discovering Prometheus pods: {e}")
        
        return None
    
    async def get_etcd_endpoints(self) -> Dict[str, Any]:
        """Get etcd endpoints from the cluster"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Look for etcd pods in openshift-etcd namespace
            etcd_namespace = "openshift-etcd"
            pods = v1.list_namespaced_pod(
                namespace=etcd_namespace,
                label_selector="app=etcd"
            )
            
            endpoints = []
            for pod in pods.items:
                if pod.status.phase == "Running":
                    pod_ip = pod.status.pod_ip
                    endpoints.append({
                        'name': pod.metadata.name,
                        'ip': pod_ip,
                        'endpoint': f"{pod_ip}:2379"
                    })
            
            return {
                'namespace': etcd_namespace,
                'endpoints': endpoints,
                'total_members': len(endpoints)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting etcd endpoints: {e}")
            return {'error': str(e)}
    
    async def execute_etcd_command(self, command: str) -> Dict[str, Any]:
        """Execute etcdctl command in etcd pod"""
        try:
            v1 = client.CoreV1Api(self.k8s_client)
            
            # Find first running etcd pod
            etcd_namespace = "openshift-etcd"
            pods = v1.list_namespaced_pod(
                namespace=etcd_namespace,
                label_selector="app=etcd"
            )
            
            if not pods.items:
                return {'error': 'No etcd pods found'}
            
            # Use first running pod
            pod_name = None
            for pod in pods.items:
                if pod.status.phase == "Running":
                    pod_name = pod.metadata.name
                    break
            
            if not pod_name:
                return {'error': 'No running etcd pods found'}
            
            # Determine container name to exec into (prefer 'etcdctl' then 'etcd')
            container_name = None
            try:
                for c in pods.items[0].spec.containers:
                    if c.name == "etcdctl":
                        container_name = c.name
                        break
                if container_name is None:
                    for c in pods.items[0].spec.containers:
                        if c.name == "etcd":
                            container_name = c.name
                            break
                if container_name is None and pods.items[0].spec.containers:
                    container_name = pods.items[0].spec.containers[0].name
            except Exception:
                pass

            # Execute command in pod
            from kubernetes.stream import stream
            
            base_cmd = [
                'etcdctl',
                '--cacert=/etc/etcd/tls/etcd-ca/ca.crt',
                '--cert=/etc/etcd/tls/etcd-peer/peer.crt', 
                '--key=/etc/etcd/tls/etcd-peer/peer.key',
                '--endpoints=https://localhost:2379'
            ] + command.split()

            # Build a shell-wrapped command to unset conflicting env vars and force ETCDCTL_API=3
            unsafe_envs = [
                'ETCDCTL_KEY','ETCDCTL_CERT','ETCDCTL_CACERT','ETCDCTL_ENDPOINTS','ETCDCTL_USER',
                'ETCDCTL_PASSWORD','ETCDCTL_TOKEN','ETCDCTL_INSECURE_SKIP_TLS_VERIFY'
            ]
            unset_parts = ' '.join([f"-u {name}" for name in unsafe_envs])
            cmd_str = ' '.join(shlex.quote(p) for p in base_cmd)
            shell_cmd = f"env {unset_parts} ETCDCTL_API=3 {cmd_str}"

            exec_command = ['/bin/sh', '-c', shell_cmd]

            resp = stream(
                v1.connect_get_namespaced_pod_exec,
                pod_name,
                etcd_namespace,
                command=exec_command,
                container=container_name,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False
            )
            
            return {
                'pod_name': pod_name,
                'command': ' '.join(exec_command),
                'output': resp
            }
            
        except Exception as e:
            self.logger.error(f"Error executing etcd command: {e}")
            return {'error': str(e)}
    
    def get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for API requests"""
        headers = {}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
            self.logger.debug("Authorization header configured")
        else:
            self.logger.warning("No token available for Authorization header")
        return headers
    
    def get_prometheus_config(self) -> Dict[str, Any]:
        """Get Prometheus connection configuration"""
        # Environment overrides
        env_url = os.getenv('PROMETHEUS_URL') or os.getenv('OCP_PROMETHEUS_URL')
        env_verify = os.getenv('PROMETHEUS_VERIFY')
        if env_url:
            self.logger.info(f"Using PROMETHEUS_URL override: {env_url}")
        base_url = env_url or self.prometheus_url
        verify_value: Any = self.ca_cert_path if self.ca_cert_path else False
        if env_verify is not None:
            # Accept 'true'/'false' or a path to CA file
            env_verify_lower = env_verify.lower()
            if env_verify_lower in ('true', '1', 'yes'):
                verify_value = True
            elif env_verify_lower in ('false', '0', 'no'):
                verify_value = False
            else:
                verify_value = env_verify  # treat as CA file path

        # Fallback URLs from env or discovery
        fallback_urls: list[str] = []
        env_fallbacks = os.getenv('PROMETHEUS_FALLBACK_URLS')
        if env_fallbacks:
            fallback_urls.extend([u.strip() for u in env_fallbacks.split(',') if u.strip()])

        fallback_urls.extend(self._build_fallback_urls())

        config = {
            'url': base_url,
            'headers': self.get_auth_headers(),
            'verify': verify_value,
            'fallback_urls': fallback_urls,
        }
        
        # Log configuration for debugging (without sensitive data)
        config_debug = config.copy()
        if 'Authorization' in config_debug.get('headers', {}):
            config_debug['headers'] = config_debug['headers'].copy()
            config_debug['headers']['Authorization'] = 'Bearer [REDACTED]'
        
        self.logger.debug(f"Prometheus config: {config_debug}")
        return config

    def _build_fallback_urls(self) -> list:
        """Construct likely fallback Prometheus URLs when the route is unreachable."""
        fallbacks: list[str] = []
        try:
            info = self.prometheus_info or {}
            namespace = info.get('namespace')
            access_method = info.get('access_method')
            service_name = info.get('service_name') or 'prometheus-k8s'
            port = info.get('port') or 9090

            # If initial access is via route, prefer cluster-internal service DNS
            if namespace:
                fallbacks.append(f"http://{service_name}.{namespace}.svc.cluster.local:{port}")
                fallbacks.append(f"http://{service_name}.{namespace}.svc:{port}")

            # Generic common namespaces if discovery context missing
            if not namespace:
                for ns in ['openshift-monitoring', 'openshift-user-workload-monitoring', 'monitoring']:
                    fallbacks.append(f"http://prometheus-k8s.{ns}.svc.cluster.local:9090")

            # If discovery returned direct pod or nodeport, include that URL too
            if access_method in ('nodeport', 'loadbalancer', 'direct_pod') and info.get('url'):
                fallbacks.append(info['url'])

        except Exception as e:
            self.logger.debug(f"Failed building fallback URLs: {e}")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for u in fallbacks:
            if u and u not in seen:
                seen.add(u)
                unique.append(u)
        return unique