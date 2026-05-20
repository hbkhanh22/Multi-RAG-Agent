"""
Router Agent - Routes queries to appropriate specialist agents
"""

from typing import Any, Dict, List
import logging
import json
import re

from .base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class RouterAgent(BaseAgent):
    """Routes queries to appropriate specialist agents"""
    
    def __init__(self):
        """Initialize router agent"""
        super().__init__(
            name="Query Router",
            model=settings.default_llm_model,
            temperature=0.0
        )
        
        self.domains = ["finance", "tech", "general"]
        self.multi_domain_threshold = 0.5
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route query to appropriate domain(s)
        
        Args:
            input_data: Must contain 'query' key
        
        Returns:
            Output dict with routing decision and confidence
        """
        try:
            query = input_data.get("query", "")
            if not query:
                return {
                    "route": "general",
                    "agents": ["general"],
                    "confidence": 1.0,
                    "reasoning": "Empty query - routing to general domain"
                }
            
            self.add_to_history("user", f"Route query: {query}")
            
            result = self._classify_query(query)
            
            self.add_to_history("assistant", json.dumps(result))
            
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
    
    def _classify_query(self, query: str) -> Dict[str, Any]:
        """
        Classify query into domains using LLM
        
        Args:
            query: Query to classify
        
        Returns:
            Classification result
        """
        try:
            system_prompt = """You are an expert query classifier. Analyze the query and classify it into one or more domains.

Domains:
- finance: stocks, investment, economics, financial reports, trading, cryptocurrencies, bonds, mutual funds, financial analysis
- tech: programming, AI, software, hardware, cloud computing, DevOps, cybersecurity, databases, web development
- general: history, science, culture, general knowledge, news, entertainment, education, sports, politics

Respond with ONLY a valid JSON object (no other text):
{
  "primary_domain": "finance|tech|general",
  "all_domains": ["finance", "tech", "general"],
  "domain_scores": {"finance": 0.8, "tech": 0.3, "general": 0.1},
  "route": "finance|tech|general|multi",
  "confidence": 0.95,
  "reasoning": "brief explanation"
}"""
            
            prompt = f"""Classify this query:

Query: "{query}"

Respond with JSON only (no markdown, no code blocks):"""
            
            response = self.invoke([prompt], system_prompt=system_prompt)
            
            # Try to extract JSON from response
            classification = self._parse_classification_response(response)
            
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
            result = {
                "primary_domain": data.get("primary_domain", "general"),
                "all_domains": data.get("all_domains", ["general"]),
                "domain_scores": data.get("domain_scores", {}),
                "route": data.get("route", "general"),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", "")
            }
            
            # Convert route and all_domains to proper format
            if result["route"] == "multi":
                # Multi-domain routing
                result["agents"] = result["all_domains"]
            else:
                result["agents"] = [result["route"]]
            
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
