import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .ollama_client import OllamaClient, ChatMessage
from .query_translator import QueryAnalysis, QueryType
from ..models.document import ProcessedChunk

logger = logging.getLogger(__name__)


class AnswerStyle(Enum):
    """Different answer generation styles"""
    CONCISE = "concise"
    DETAILED = "detailed"
    TECHNICAL = "technical"
    BEGINNER_FRIENDLY = "beginner_friendly"


@dataclass
class AnswerGenerationConfig:
    """Configuration for answer generation"""
    style: AnswerStyle = AnswerStyle.DETAILED
    max_answer_length: int = 1000
    include_sources: bool = True
    include_confidence: bool = True
    cite_sections: bool = True
    explain_technical_terms: bool = False


@dataclass
class GeneratedAnswer:
    """Generated answer with metadata"""
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    reasoning: str
    query_analysis: QueryAnalysis
    context_used: int
    generation_config: AnswerGenerationConfig


class AnswerGenerator:
    """
    Generates comprehensive answers using retrieved context
    and query analysis for optimal response formatting
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        logger.info("Answer generator initialized")
    
    def generate_answer(self,
                       query_analysis: QueryAnalysis,
                       context_chunks: List[ProcessedChunk],
                       chat_history: List[ChatMessage] = None,
                       config: AnswerGenerationConfig = None) -> GeneratedAnswer:
        """
        Generate comprehensive answer using query analysis and retrieved context
        
        Args:
            query_analysis: Analysis of the user's query
            context_chunks: Retrieved context chunks
            chat_history: Previous conversation context
            config: Answer generation configuration
            
        Returns:
            GeneratedAnswer with response and metadata
        """
        if config is None:
            config = AnswerGenerationConfig()
        
        try:
            # Build context string from chunks
            context_str = self._build_context_string(context_chunks, config)
            
            # Generate answer using LLM
            answer = self._generate_llm_response(
                query_analysis, context_str, chat_history, config
            )
            
            # Extract sources information
            sources = self._extract_sources(context_chunks)
            
            # Generate reasoning explanation
            reasoning = self._generate_reasoning(query_analysis, context_chunks)
            
            # Calculate confidence
            confidence = self._calculate_answer_confidence(
                query_analysis, context_chunks, answer
            )
            
            generated_answer = GeneratedAnswer(
                answer=answer,
                confidence=confidence,
                sources=sources,
                reasoning=reasoning,
                query_analysis=query_analysis,
                context_used=len(context_chunks),
                generation_config=config
            )
            
            logger.info(f"Answer generated: {len(answer)} chars, confidence={confidence:.2f}")
            return generated_answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            # Return fallback answer
            return GeneratedAnswer(
                answer=f"I apologize, but I encountered an error while generating the answer: {str(e)}",
                confidence=0.0,
                sources=[],
                reasoning="Error occurred during generation",
                query_analysis=query_analysis,
                context_used=len(context_chunks),
                generation_config=config
            )
    
    def _build_context_string(self, chunks: List[ProcessedChunk], config: AnswerGenerationConfig) -> str:
        """Build formatted context string from chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Create section reference
            section_ref = ""
            if config.cite_sections and chunk.section_header != "Document Root":
                section_ref = f"[Section: {chunk.section_header}] "
            
            # Create context entry
            context_entry = f"""[CONTEXT {i}]
{section_ref}{chunk.content.strip()}
"""
            context_parts.append(context_entry)
        
        return "\n".join(context_parts)
    
    def _generate_llm_response(self,
                              query_analysis: QueryAnalysis,
                              context_str: str,
                              chat_history: List[ChatMessage] = None,
                              config: AnswerGenerationConfig = None) -> str:
        """Generate the actual LLM response"""
        
        # Build conversation context
        conversation_context = ""
        if chat_history:
            recent_messages = chat_history[-3:]
            conversation_context = "\n".join([
                f"{msg.role}: {msg.content}" for msg in recent_messages
            ])
        
        # Create style-specific instructions
        style_instructions = self._get_style_instructions(config.style, query_analysis.query_type)
        
        # Build comprehensive prompt
        prompt = self._build_generation_prompt(
            query_analysis,
            context_str,
            conversation_context,
            style_instructions,
            config
        )
        
        # Generate response
        messages = [ChatMessage(role="user", content=prompt)]
        response = self.llm_client.chat(messages)
        
        return response.content.strip()
    
    def _get_style_instructions(self, style: AnswerStyle, query_type: QueryType) -> str:
        """Get style-specific instructions for answer generation"""
        
        base_instructions = {
            AnswerStyle.CONCISE: "Provide a concise, direct answer focusing on the essential information.",
            AnswerStyle.DETAILED: "Provide a comprehensive, detailed explanation with context and examples.",
            AnswerStyle.TECHNICAL: "Use precise technical language and include relevant specifications and details.",
            AnswerStyle.BEGINNER_FRIENDLY: "Explain in simple terms, defining technical concepts and providing context."
        }
        
        query_specific = {
            QueryType.FACTUAL: "Focus on providing accurate facts and specific information.",
            QueryType.EXPLANATORY: "Explain the concepts clearly with reasoning and mechanisms.",
            QueryType.PROCEDURAL: "Provide clear step-by-step instructions or processes.",
            QueryType.COMPARATIVE: "Compare and contrast the different options clearly.",
            QueryType.TROUBLESHOOTING: "Focus on practical solutions and troubleshooting steps.",
            QueryType.SPECIFICATION: "Provide precise technical specifications and standards."
        }
        
        return f"{base_instructions.get(style, '')} {query_specific.get(query_type, '')}"
    
    def _build_generation_prompt(self,
                                query_analysis: QueryAnalysis,
                                context_str: str,
                                conversation_context: str,
                                style_instructions: str,
                                config: AnswerGenerationConfig) -> str:
        """Build the comprehensive prompt for answer generation"""
        
        prompt = f"""You are an expert in NVMe (Non-Volatile Memory Express) technology and PCIe storage systems. Your task is to answer the user's question using the provided technical documentation context.

QUERY ANALYSIS:
- Original Question: {query_analysis.original_query}
- Query Type: {query_analysis.query_type.value}
- Intent: {query_analysis.intent}
- Technical Terms: {', '.join(query_analysis.technical_terms)}

STYLE INSTRUCTIONS:
{style_instructions}

CONVERSATION CONTEXT:
{conversation_context if conversation_context else "None"}

TECHNICAL DOCUMENTATION CONTEXT:
{context_str}

ANSWER REQUIREMENTS:
1. Answer the question directly and accurately using the provided context
2. Base your response strictly on the information provided in the context
3. If the context doesn't contain enough information, acknowledge this limitation
4. Use precise technical terminology from the NVMe specification
5. {"Include section references where relevant" if config.cite_sections else ""}
6. {"Explain technical terms for clarity" if config.explain_technical_terms else ""}
7. Keep the answer focused and relevant to the specific question
8. Maximum length: approximately {config.max_answer_length} words

IMPORTANT: Only use information from the provided context. Do not add information not present in the context.

ANSWER:"""

        return prompt
    
    def _extract_sources(self, chunks: List[ProcessedChunk]) -> List[Dict[str, Any]]:
        """Extract source information from context chunks"""
        sources = []
        
        for chunk in chunks:
            source_info = {
                "section": chunk.section_header,
                "page": chunk.page_number,
                "document_id": chunk.parent_doc_id,
                "chunk_id": chunk.chunk_id,
                "content_preview": chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
            }
            sources.append(source_info)
        
        return sources
    
    def _generate_reasoning(self, query_analysis: QueryAnalysis, chunks: List[ProcessedChunk]) -> str:
        """Generate explanation of the reasoning process"""
        
        sections_used = list(set(chunk.section_header for chunk in chunks))
        
        reasoning = f"""The answer was generated based on the following analysis:

Query Analysis:
- Type: {query_analysis.query_type.value}
- Intent: {query_analysis.intent}
- Confidence: {query_analysis.confidence:.2f}

Context Used:
- {len(chunks)} relevant chunks from the documentation
- Sections covered: {', '.join(sections_used)}
- Technical terms identified: {', '.join(query_analysis.technical_terms)}

The response synthesizes information from these sources to provide a comprehensive answer to the user's question."""

        return reasoning
    
    def _calculate_answer_confidence(self,
                                   query_analysis: QueryAnalysis,
                                   chunks: List[ProcessedChunk],
                                   answer: str) -> float:
        """Calculate confidence score for the generated answer"""
        
        # Start with query analysis confidence
        confidence = query_analysis.confidence
        
        # Adjust based on context quality
        if len(chunks) >= 3:
            confidence += 0.1
        elif len(chunks) == 0:
            confidence = 0.1
        
        # Adjust based on answer length and detail
        if len(answer) > 100:  # Substantial answer
            confidence += 0.1
        
        # Check if answer acknowledges limitations
        if any(phrase in answer.lower() for phrase in [
            "not enough information", "context doesn't contain",
            "insufficient detail", "not specified"
        ]):
            confidence = max(0.3, confidence - 0.2)  # Lower but not too low for honest answers
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, confidence))
    
    def generate_follow_up_questions(self,
                                   query_analysis: QueryAnalysis,
                                   generated_answer: GeneratedAnswer) -> List[str]:
        """Generate relevant follow-up questions"""
        try:
            follow_up_prompt = f"""Based on this NVMe technical question and answer, suggest 3 relevant follow-up questions that a user might want to ask to deepen their understanding.

Original Question: {query_analysis.original_query}
Answer: {generated_answer.answer[:300]}...

Generate 3 specific, technical follow-up questions (one per line, no numbering):"""

            messages = [ChatMessage(role="user", content=follow_up_prompt)]
            response = self.llm_client.chat(messages)
            
            # Parse follow-up questions
            questions = []
            for line in response.content.strip().split('\n'):
                line = line.strip()
                if line and '?' in line:
                    # Remove any numbering or bullets
                    import re
                    cleaned = re.sub(r'^[\d\.\-\*\+]\s*', '', line)
                    if cleaned:
                        questions.append(cleaned)
            
            return questions[:3]  # Return up to 3 questions
            
        except Exception as e:
            logger.warning(f"Follow-up question generation failed: {e}")
            return []
    
    def update_config(self, **kwargs):
        """Update answer generation configuration"""
        # This would update default configuration
        pass