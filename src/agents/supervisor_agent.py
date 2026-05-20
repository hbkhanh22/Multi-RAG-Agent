"""
Supervisor Agent - Orchestrates multiple specialist agents for complex queries
"""

from typing import Any, Dict, List
import logging
import asyncio

from .base_agent import BaseAgent
from .rag_agent import RAGAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class SupervisorAgent(BaseAgent):
    """Coordinates multiple specialist RAG agents"""
    
    def __init__(self):
        """Initialize supervisor agent"""
        super().__init__(
            name="Supervisor Agent",
            model=settings.powerful_llm_model,
            temperature=0.1
        )
        
        # Initialize specialist agents
        self.agents = {
            "finance": RAGAgent("finance"),
            "tech": RAGAgent("tech"),
            "general": RAGAgent("general")
        }
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple agents for complex query
        
        Args:
            input_data: Must contain 'query' and 'active_agents' keys
        
        Returns:
            Coordinated response from all active agents
        """
        try:
            query = input_data.get("query", "")
            active_agents = input_data.get("active_agents", ["general"])
            
            if not query:
                return {
                    "success": False,
                    "answer": "No query provided",
                    "agent_responses": {},
                    "sources": []
                }
            
            self.add_to_history("user", query)
            
            # Step 1: Create execution plan
            plan = self._create_execution_plan(query, active_agents)
            logger.info(f"Created execution plan: {plan}")
            
            # Step 2: Execute agents in parallel
            agent_responses = self._execute_agents_parallel(query, active_agents)
            
            # Step 3: Synthesize responses
            synthesized_answer = self._synthesize_responses(query, agent_responses)
            
            self.add_to_history("assistant", synthesized_answer)
            
            return {
                "success": True,
                "answer": synthesized_answer,
                "agent_responses": agent_responses,
                "execution_plan": plan,
                "sources": self._extract_all_citations(agent_responses)
            }
        
        except Exception as e:
            logger.error(f"Error in supervisor: {e}")
            return {
                "success": False,
                "answer": f"Error in multi-agent coordination: {str(e)}",
                "agent_responses": {},
                "sources": [],
                "error": str(e)
            }
    
    def _create_execution_plan(self, query: str, agents: List[str]) -> Dict[str, Any]:
        """
        Create execution plan for multi-agent coordination
        
        Args:
            query: User query
            agents: List of agents to coordinate
        
        Returns:
            Execution plan
        """
        try:
            system_prompt = """You are coordinating multiple specialist agents.
Create a clear execution plan for gathering comprehensive information."""
            
            agent_list = ", ".join([a.upper() for a in agents])
            prompt = f"""Query: {query}

Available agents: {agent_list}

Create a brief plan for how each agent should approach this query.
Focus on what each agent should investigate."""
            
            plan_text = self.invoke([prompt], system_prompt=system_prompt)
            
            return {
                "agents": agents,
                "strategy": plan_text,
                "parallel_execution": True
            }
        
        except Exception as e:
            logger.error(f"Error creating plan: {e}")
            return {
                "agents": agents,
                "strategy": "Execute all agents in parallel",
                "parallel_execution": True,
                "error": str(e)
            }
    
    def _execute_agents_parallel(self, query: str, active_agents: List[str]) -> Dict[str, Dict]:
        """
        Execute multiple agents in parallel
        
        Args:
            query: User query
            active_agents: List of agent names to execute
        
        Returns:
            Dict mapping agent names to their responses
        """
        responses = {}
        
        try:
            # Execute agents sequentially (simplified parallel approach)
            for agent_name in active_agents:
                if agent_name not in self.agents:
                    logger.warning(f"Unknown agent: {agent_name}")
                    continue
                
                try:
                    agent = self.agents[agent_name]
                    result = agent.run({"query": query})
                    responses[agent_name] = result
                    logger.info(f"{agent_name} agent completed")
                except Exception as e:
                    logger.error(f"Error executing {agent_name} agent: {e}")
                    responses[agent_name] = {
                        "success": False,
                        "error": str(e),
                        "answer": f"Error in {agent_name} agent"
                    }
            
            return responses
        
        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            return {"error": str(e)}
    
    def _synthesize_responses(self, query: str, responses: Dict[str, Dict]) -> str:
        """
        Synthesize responses from multiple agents
        
        Args:
            query: Original query
            responses: Dict of agent responses
        
        Returns:
            Synthesized answer
        """
        try:
            # Format agent responses
            formatted_responses = self._format_agent_responses(responses)
            
            system_prompt = """You are synthesizing information from multiple specialist agents.
Create a comprehensive, unified answer that integrates insights from all sources.
- Merge related information
- Remove redundancy
- Maintain source attribution
- Flag any contradictions
- Present in clear, structured format"""
            
            prompt = f"""Original Query: {query}

Agent Responses:
{formatted_responses}

Please synthesize a comprehensive answer combining all perspectives:"""
            
            synthesized = self.invoke([prompt], system_prompt=system_prompt)
            return synthesized
        
        except Exception as e:
            logger.error(f"Error synthesizing responses: {e}")
            return self._create_fallback_synthesis(responses)
    
    def _format_agent_responses(self, responses: Dict[str, Dict]) -> str:
        """Format agent responses for LLM synthesis"""
        formatted = ""
        for agent_name, response in responses.items():
            if response.get("success"):
                formatted += f"\n[{agent_name.upper()}]\n"
                formatted += f"Answer: {response.get('answer', 'No answer')}\n"
            else:
                formatted += f"\n[{agent_name.upper()}] - ERROR\n"
                formatted += f"Error: {response.get('error', 'Unknown error')}\n"
        return formatted
    
    def _create_fallback_synthesis(self, responses: Dict[str, Dict]) -> str:
        """Create fallback synthesis if LLM synthesis fails"""
        synthesis = "## Synthesized Answer from Multiple Agents\n\n"
        
        for agent_name, response in responses.items():
            if response.get("success"):
                synthesis += f"### {agent_name.capitalize()}\n"
                synthesis += f"{response.get('answer', 'No answer')}\n\n"
        
        return synthesis
    
    def _extract_all_citations(self, responses: Dict[str, Dict]) -> List[Dict]:
        """Extract citations from all agent responses"""
        citations = []
        
        for agent_name, response in responses.items():
            if response.get("success"):
                agent_citations = response.get("sources", [])
                for citation in agent_citations:
                    citation["agent"] = agent_name
                    citations.append(citation)
        
        return citations
    
    def get_supervision_status(self) -> Dict[str, Any]:
        """Get status of supervisor and all agents"""
        status = {
            "supervisor": self.get_agent_info(),
            "agents": {}
        }
        
        for agent_name, agent in self.agents.items():
            status["agents"][agent_name] = agent.get_agent_info()
        
        return status
