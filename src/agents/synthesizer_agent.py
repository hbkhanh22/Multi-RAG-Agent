"""
Synthesizer Agent - Combines and synthesizes final answers from all sources
"""

from typing import Any, Dict, List
import logging

from .base_agent import BaseAgent
from config.settings import settings

logger = logging.getLogger(__name__)


class SynthesizerAgent(BaseAgent):
    """Synthesizes final answers from multiple sources"""
    
    def __init__(self):
        """Initialize synthesizer agent"""
        super().__init__(
            name="Answer Synthesizer",
            model=settings.default_llm_model,
            temperature=0.0
        )
    
    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize final answer from multiple sources
        
        Args:
            input_data: Must contain 'query' and 'agent_responses' keys
        
        Returns:
            Synthesized answer with metadata
        """
        try:
            query = input_data.get("query", "")
            agent_responses = input_data.get("agent_responses", {})
            
            if not agent_responses:
                return {
                    "success": False,
                    "synthesized_answer": "No agent responses to synthesize",
                    "sources": []
                }
            
            self.add_to_history("user", f"Synthesize responses for: {query}")
            
            # Step 1: Extract content from all sources
            source_content = self._extract_source_content(agent_responses)
            
            # Step 2: Identify key themes
            themes = self._identify_themes(query, source_content)
            
            # Step 3: Synthesize final answer
            final_answer = self._synthesize_content(query, source_content, themes)
            
            # Step 4: Extract citations
            citations = self._extract_citations(agent_responses)
            
            # Step 5: Create output
            output = {
                "success": True,
                "query": query,
                "synthesized_answer": final_answer,
                "key_themes": themes,
                "sources": citations,
                "source_count": len(citations),
                "agent_count": len(agent_responses)
            }
            
            self.add_to_history("assistant", final_answer)
            
            return output
        
        except Exception as e:
            logger.error(f"Error in synthesizer: {e}")
            return {
                "success": False,
                "synthesized_answer": f"Error in synthesis: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def _extract_source_content(self, agent_responses: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract answer content from all agent responses"""
        sources = {}
        
        for agent_name, response in agent_responses.items():
            if isinstance(response, dict) and response.get("success"):
                answer = response.get("answer", "")
                if answer:
                    sources[agent_name] = answer
        
        return sources
    
    def _identify_themes(self, query: str, source_content: Dict[str, str]) -> List[str]:
        """Identify key themes across all sources"""
        try:
            if not source_content:
                return []
            
            # Prepare content summary for analysis
            content_summary = "\n\n".join([
                f"[{k}]: {v[:500]}"  # Limit to first 500 chars
                for k, v in source_content.items()
            ])
            
            system_prompt = """Extract 3-5 key themes or main points from the provided responses.
These should be the central ideas or topics that appear across the content.
Return as a JSON array of strings."""
            
            prompt = f"""Query: {query}

Content from multiple sources:
{content_summary}

Extract the key themes:"""
            
            response = self.invoke([prompt], system_prompt=system_prompt)
            
            # Try to parse JSON
            import json
            try:
                themes = json.loads(response)
                if isinstance(themes, list):
                    return themes[:5]  # Limit to 5 themes
            except json.JSONDecodeError:
                # Extract themes from text
                lines = response.split("\n")
                themes = [line.strip("- ").strip() for line in lines if line.strip().startswith("-")]
                return themes[:5]
        
        except Exception as e:
            logger.error(f"Error identifying themes: {e}")
            return []
    
    def _synthesize_content(self, query: str, source_content: Dict[str, str], themes: List[str]) -> str:
        """Synthesize content into coherent final answer"""
        try:
            # Format sources for LLM
            formatted_sources = self._format_sources_for_synthesis(source_content)
            
            themes_str = "\n".join([f"- {theme}" for theme in themes]) if themes else "No identified themes"
            
            system_prompt = """You are an expert at synthesizing information from multiple sources.
Your task is to create a comprehensive, well-organized answer that:
1. Integrates insights from all sources
2. Eliminates redundancy
3. Maintains logical flow and clarity
4. Preserves source attributions
5. Flags any contradictions
6. Uses clear formatting and structure"""
            
            prompt = f"""Original Query: {query}

Key Themes Identified:
{themes_str}

Content from Multiple Sources:
{formatted_sources}

Please create a comprehensive synthesized answer that integrates all these perspectives.
Use [Source: agency] notation to attribute information.
Organize the answer by theme or topic for clarity."""
            
            synthesized = self.invoke([prompt], system_prompt=system_prompt)
            return synthesized
        
        except Exception as e:
            logger.error(f"Error synthesizing content: {e}")
            # Fallback: simple concatenation
            return self._simple_synthesis(source_content)
    
    def _format_sources_for_synthesis(self, source_content: Dict[str, str]) -> str:
        """Format source content for LLM synthesis"""
        formatted = ""
        for source_name, content in source_content.items():
            formatted += f"\n[Source: {source_name.upper()}]\n"
            formatted += f"{content[:1000]}\n"  # Limit to 1000 chars per source
            formatted += "-" * 80 + "\n"
        return formatted
    
    def _simple_synthesis(self, source_content: Dict[str, str]) -> str:
        """Simple synthesis when LLM synthesis fails"""
        synthesis = "## Answer from Multiple Sources\n\n"
        
        for source_name, content in source_content.items():
            synthesis += f"### {source_name.capitalize()}\n"
            synthesis += f"{content}\n\n"
        
        return synthesis
    
    def _extract_citations(self, agent_responses: Dict[str, Any]) -> List[Dict]:
        """Extract all citations from agent responses"""
        citations = []
        
        for agent_name, response in agent_responses.items():
            if isinstance(response, dict) and response.get("success"):
                agent_sources = response.get("sources", [])
                
                for source in agent_sources:
                    if isinstance(source, dict):
                        citation = {
                            "agent": agent_name,
                            "source_id": source.get("source_id"),
                            "source": source.get("source"),
                            "page": source.get("page")
                        }
                        citations.append(citation)
                
                # Add metadata about retrieval
                citations.append({
                    "agent": agent_name,
                    "retrieved_count": response.get("retrieved_count", 0),
                    "reranked_count": response.get("reranked_count", 0),
                    "type": "metadata"
                })
        
        return citations
    
    def create_structured_output(self, synthesis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured output with all metadata"""
        return {
            "answer": synthesis_result.get("synthesized_answer", ""),
            "query": synthesis_result.get("query", ""),
            "themes": synthesis_result.get("key_themes", []),
            "citations": synthesis_result.get("sources", []),
            "metadata": {
                "synthesizer_model": self.model,
                "total_agents": synthesis_result.get("agent_count", 0),
                "total_sources": synthesis_result.get("source_count", 0),
                "synthesis_status": "success" if synthesis_result.get("success") else "failed"
            }
        }
