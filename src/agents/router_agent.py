"""
Router Agent - Routes queries to appropriate specialist agents
"""

from typing import Any, Dict, List
import logging
import json
import re
import unicodedata

from .base_agent import BaseAgent
from config.settings import settings
from src.mcp_client import mcp_manager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- fast path
# Pure small talk never needs retrieval, so it must not wait for an LLM call to be
# classified. Matching is on the whole message (accents/punctuation ignored).
_SMALLTALK = re.compile(
    r"^(?:"
    r"xin chao|chao|chao buoi (?:sang|trua|chieu|toi)|hello|hi|hey|hola|alo|yo|"
    r"good (?:morning|afternoon|evening|night)|"
    r"cam on|thanks|thank you|thank u|cam on ban|"
    r"tam biet|bye|goodbye|see you|hen gap lai|"
    r"ok|oke|okay|duoc roi|tuyet voi|hay qua|"
    r"ban khoe khong|khoe khong|how are you|"
    r"ban la ai|ban ten gi|who are you|what is your name|"
    r"ban (?:co the |giup duoc )?lam (?:duoc )?gi|what can you do|ban giup duoc gi"
    r")(?:\s+(?:a|ah|nhe|nha|nhen|ban|oi|bot|ai|assistant|rat|nhieu|nhe|em|anh|chi|ban oi|ha|va|!))*$"
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def fast_route(query: str):
    """Return a direct_chat routing decision for obvious small talk, else None."""
    if len(query) <= 60 and _SMALLTALK.match(_normalize(query)):
        return {
            "primary_domain": "direct_chat",
            "all_domains": ["direct_chat"],
            "domain_scores": {"direct_chat": 1.0},
            "route": "direct_chat",
            "agents": ["direct_chat"],
            "confidence": 1.0,
            "reasoning": "Small talk (fast path, no LLM)",
        }
    return None


class RouterAgent(BaseAgent):
    """Routes queries to appropriate specialist agents"""
    
    def __init__(self):
        """Initialize router agent"""
        super().__init__(
            name="Query Router",
            model=settings.router_llm_model or settings.default_llm_model,
            temperature=0.0,
            max_tokens=settings.router_max_tokens,
        )
        
        self.domains = ["finance", "tech", "general", "web", "tools"]
        self.multi_domain_threshold = 0.5
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route query to appropriate domain(s). All classification is done by LLM.
        
        Args:
            input_data: Must contain 'query' key
        
        Returns:
            Output dict with routing decision and confidence
        """
        try:
            query = input_data.get("query", "").strip()
            if not query:
                return {
                    "primary_domain": "direct_chat",
                    "all_domains": ["direct_chat"],
                    "domain_scores": {"direct_chat": 1.0},
                    "route": "direct_chat",
                    "agents": ["direct_chat"],
                    "confidence": 1.0,
                    "reasoning": "Empty query — handled as direct chat"
                }
            
            result = fast_route(query) or self._classify_query(query)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in router: {e}")
            return {
                "route": "general",
                "agents": ["general"],
                "confidence": 0.5,
                "reasoning": f"Error in routing: {str(e)}",
                "error": str(e)
            }
    
    @staticmethod
    def _tools_available() -> bool:
        return mcp_manager.has_tools()

    def _fill_tools(self, prompt: str) -> str:
        """Offer the 'tools' route only while MCP servers are actually connected."""
        domains = ["finance", "tech", "general", "web"] + (["tools"] if self._tools_available() else []) + ["direct_chat"]
        tools_rule = tools_key = ""
        if self._tools_available():
            tools_rule = (
                "- tools: The user explicitly asks to read, list or search FILES in the workspace folder, "
                "or to open/fetch/summarize the content of a SPECIFIC URL or web page.\n"
                '  Examples: "Đọc file hello.txt", "Liệt kê các file trong workspace", "Tóm tắt trang https://example.com"\n'
            )
            tools_key = "4. Use 'tools' ONLY for file access or fetching a given URL, not for general web search.\n"
        return (prompt.replace("{tools_rule}", tools_rule).replace("{tools_key_rule}", tools_key)
                .replace("{domain_enum}", "|".join(domains))
                .replace("{domain_list}", ", ".join(f'"{d}"' for d in domains)))

    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query into domains using LLM
        
        Args:
            query: Query to classify
        
        Returns:
            Classification result
        """
        try:
            system_prompt = """You are a query router for an intelligent Multi-RAG & Autonomous Agent system.
Your task is to analyze the user's message and determine the optimal routing:
- direct_chat: Casual conversation, greetings, meta-questions about the AI, no external information needed.
  Examples: "Hello!", "Xin chào", "Cảm ơn bạn", "Bạn là ai?", "Bạn làm được gì?", "Tạm biệt"
- web: Real-time queries, current news, live updates, today's events, weather, stock quotes today, latest software versions, or explicit requests to search online / internet / Google.
  Examples: "Tin tức mới nhất hôm nay", "Giá vàng / Bitcoin hôm nay bao nhiêu?", "Tìm kiếm web về...", "Thời tiết Hà Nội hôm nay"
- finance: In-depth financial analysis, corporate filings, balance sheets, economics, bond markets, banking regulations, and historical financial concepts stored in internal docs.
- tech: Programming languages, AI/ML theory, system architecture, backend/frontend engineering, DevOps, internal tech documentations.
- general: General knowledge, history, geography, encyclopedic topics stored in internal docs.
{tools_rule}- multi: Complex queries requiring information from multiple domains (e.g. tech + finance, or finance + web).

Key rules:
1. ONLY choose direct_chat when there is NO lookup needed.
2. If the user asks about current events, breaking news, "hôm nay", "mới nhất", "bây giờ", or requests web search, choose 'web'.
3. If the query requires both domain docs and web search, choose 'multi' with both in all_domains.

{tools_key_rule}Respond with ONLY a valid JSON object, no markdown, no explanation outside the JSON:
{
  "primary_domain": "{domain_enum}",
  "all_domains": [{domain_list}],
  "domain_scores": {"finance": 0.0, "tech": 0.0, "general": 0.0, "web": 0.0, "direct_chat": 0.0},
  "route": "{domain_enum}|multi",
  "confidence": 0.95,
  "reasoning": "one-sentence explanation"
}"""
            system_prompt = self._fill_tools(system_prompt)
            
            prompt = f"""Classify this query:

Query: "{query}"

Respond with JSON only:"""
            
            response = self.invoke([prompt], system_prompt=system_prompt)
            
            # Try to extract JSON from response
            classification = self._parse_classification_response(response)

            # 'tools' without connected MCP servers cannot be served: use web search instead
            if "tools" in classification.get("agents", []) and not self._tools_available():
                agents = [("web" if a == "tools" else a) for a in classification["agents"]]
                classification["agents"] = list(dict.fromkeys(agents))
                classification["all_domains"] = classification["agents"]
                classification["route"] = (
                    classification["agents"][0] if len(classification["agents"]) == 1 else classification["route"]
                )

            return classification
        
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            # Fallback to general
            return {
                "primary_domain": "general",
                "all_domains": ["general"],
                "route": "general",
                "agents": ["general"],
                "confidence": 0.5,
                "reasoning": "Fallback classification due to error",
                "error": str(e)
            }
    
    def _parse_classification_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract classification
        
        Args:
            response: LLM response text
        
        Returns:
            Parsed classification dict
        """
        try:
            # Try direct JSON parsing
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try extracting JSON from markdown code block
                json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    # Try finding JSON object pattern
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No JSON found in response")
            
            # Validate and normalize response
            valid_domains = [d for d in self.domains]
            route = str(data.get("route", "general")).strip().lower()
            all_domains = data.get("all_domains") or []
            if not isinstance(all_domains, list):
                all_domains = []

            if route == "multi":
                agents = [d for d in all_domains if d in valid_domains]
                agents = list(dict.fromkeys(agents))  # dedupe, keep order
                if not agents:
                    route, agents = "general", ["general"]
                elif len(agents) == 1:
                    route = agents[0]
            elif route in valid_domains or route == "direct_chat":
                agents = [route]
            else:
                route, agents = "general", ["general"]

            try:
                confidence = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5

            result = {
                "primary_domain": data.get("primary_domain", route),
                "all_domains": agents,
                "domain_scores": data.get("domain_scores", {}),
                "route": route,
                "agents": agents,
                "confidence": confidence,
                "reasoning": data.get("reasoning", ""),
            }
            
            logger.info(f"Classified to: {result['route']} with confidence {result['confidence']}")
            return result
        
        except Exception as e:
            logger.error(f"Error parsing classification: {e}")
            # Fallback
            return {
                "primary_domain": "general",
                "all_domains": ["general"],
                "route": "general",
                "agents": ["general"],
                "confidence": 0.5,
                "reasoning": "Parsing error - defaulting to general"
            }
    
    def route_to_agents(self, query: str) -> tuple:
        """
        Route query and return list of agents to invoke
        
        Args:
            query: User query
        
        Returns:
            Tuple of (route_type, agent_list, confidence)
        """
        result = self.run({"query": query})
        
        route = result.get("route", "general")
        agents = result.get("agents", ["general"])
        confidence = result.get("confidence", 0.5)
        
        return route, agents, confidence
    
    def explain_routing(self, query: str) -> str:
        """Get explanation of routing decision"""
        result = self.run({"query": query})
        
        explanation = f"""
Routing Decision for Query: "{query}"

Primary Domain: {result.get('primary_domain', 'unknown').upper()}
Route Type: {result.get('route', 'unknown').upper()}
Agents to Invoke: {', '.join(result.get('agents', ['general']))}
Confidence: {result.get('confidence', 0):.1%}

Reasoning: {result.get('reasoning', 'No reasoning provided')}

Domain Scores:
"""
        scores = result.get("domain_scores", {})
        for domain, score in scores.items():
            explanation += f"  - {domain}: {score:.1%}\n"
        
        return explanation
