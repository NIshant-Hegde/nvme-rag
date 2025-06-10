import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from .ollama_client import OllamaClient, ChatMessage

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"
    EXPLANATORY = "explanatory"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    TROUBLESHOOTING = "troubleshooting"
    SPECIFICATION = "specification"


@dataclass
class QueryAnalysis:
    """Analysis of a user query"""
    original_query: str
    query_type: QueryType
    technical_terms: List[str]
    intent: str
    expanded_query: str
    search_keywords: List[str]
    confidence: float


class QueryTranslator:
    """
    Translates natural language queries into optimized search queries
    and analyzes query intent for better answer generation
    """
    
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client
        
        # Technical domain terms for NVMe
        self.nvme_terms = {
            "sq": "submission queue",
            "cq": "completion queue",
            "cmd": "command",
            "dma": "direct memory access",
            "pcie": "peripheral component interconnect express",
            "ssd": "solid state drive",
            "lba": "logical block address",
            "prp": "physical region page",
            "sgl": "scatter gather list",
            "nsid": "namespace identifier",
            "cdw": "command dword",
            "cid": "command identifier",
            "sq": "submission queue",
            "cq": "completion queue"
        }
        
        logger.info("Query translator initialized")
    
    def analyze_query(self, query: str, chat_history: List[ChatMessage] = None) -> QueryAnalysis:
        """
        Analyze query to understand intent and optimize for retrieval
        
        Args:
            query: User's natural language query
            chat_history: Previous conversation context
            
        Returns:
            QueryAnalysis with query understanding and optimization
        """
        try:
            # Step 1: Analyze query type and intent
            query_type, intent = self._classify_query(query)
            
            # Step 2: Extract and expand technical terms
            technical_terms = self._extract_technical_terms(query)
            
            # Step 3: Generate expanded query for better retrieval
            expanded_query = self._expand_query(query, chat_history)
            
            # Step 4: Generate search keywords
            search_keywords = self._generate_search_keywords(query, technical_terms)
            
            # Step 5: Calculate confidence
            confidence = self._calculate_confidence(query, technical_terms)
            
            analysis = QueryAnalysis(
                original_query=query,
                query_type=query_type,
                technical_terms=technical_terms,
                intent=intent,
                expanded_query=expanded_query,
                search_keywords=search_keywords,
                confidence=confidence
            )
            
            logger.info(f"Query analyzed: type={query_type.value}, confidence={confidence:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Query analysis failed: {e}")
            # Return minimal analysis as fallback
            return QueryAnalysis(
                original_query=query,
                query_type=QueryType.FACTUAL,
                technical_terms=[],
                intent="general information request",
                expanded_query=query,
                search_keywords=query.split(),
                confidence=0.5
            )
    
    def _classify_query(self, query: str) -> tuple[QueryType, str]:
        """Classify the type of query and extract intent"""
        try:
            classification_prompt = f"""Analyze this technical query about NVMe (Non-Volatile Memory Express) and determine:
1. The query type from these categories:
   - factual: asking for specific facts, definitions, or values
   - explanatory: asking how something works or why
   - procedural: asking how to do something or steps
   - comparative: comparing different concepts or options
   - troubleshooting: seeking solutions to problems
   - specification: asking about technical specifications or standards

2. The user's intent in 1-2 sentences

Query: "{query}"

Respond in exactly this format:
Type: [query_type]
Intent: [intent description]"""

            messages = [ChatMessage(role="user", content=classification_prompt)]
            response = self.llm_client.chat(messages)
            
            # Parse response
            lines = response.content.strip().split('\n')
            query_type = QueryType.FACTUAL  # default
            intent = "general information request"
            
            for line in lines:
                if line.startswith("Type:"):
                    type_str = line.split(":", 1)[1].strip().lower()
                    try:
                        query_type = QueryType(type_str)
                    except ValueError:
                        pass
                elif line.startswith("Intent:"):
                    intent = line.split(":", 1)[1].strip()
            
            return query_type, intent
            
        except Exception as e:
            logger.warning(f"Query classification failed: {e}")
            return QueryType.FACTUAL, "general information request"
    
    def _extract_technical_terms(self, query: str) -> List[str]:
        """Extract and expand technical terms from the query"""
        terms = []
        query_lower = query.lower()
        
        # Look for known abbreviations and expand them
        for abbrev, full_term in self.nvme_terms.items():
            if abbrev in query_lower:
                terms.append(full_term)
                terms.append(abbrev.upper())  # Add uppercase version too
        
        # Extract potential technical terms (typically 2-4 character combinations or camelCase)
        import re
        
        # Look for acronyms (2-5 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,5}\b', query)
        terms.extend(acronyms)
        
        # Look for technical patterns
        tech_patterns = re.findall(r'\b(?:queue|command|memory|address|block|page|namespace|controller|submission|completion)\b', query_lower)
        terms.extend(tech_patterns)
        
        return list(set(terms))  # Remove duplicates
    
    def _expand_query(self, query: str, chat_history: List[ChatMessage] = None) -> str:
        """Expand query with context and technical details for better retrieval"""
        try:
            context_str = ""
            if chat_history:
                recent_messages = chat_history[-3:]
                context_str = "\n".join([f"{msg.role}: {msg.content}" for msg in recent_messages])
            
            expansion_prompt = f"""You are an expert in NVMe (Non-Volatile Memory Express) technology. 
Expand this query to include relevant technical terms and context that would help find the most relevant information from technical documentation.

Guidelines:
- Add relevant technical synonyms and related terms
- Expand abbreviations when helpful
- Include context that clarifies the question
- Keep the core intent unchanged
- Make it more specific and searchable
- Focus on NVMe, PCIe, storage, and related technologies

Conversation Context:
{context_str}

Original Query: {query}

Expanded Query (return only the expanded query):"""

            messages = [ChatMessage(role="user", content=expansion_prompt)]
            response = self.llm_client.chat(messages)
            
            expanded = response.content.strip()
            
            # Fallback to original if expansion failed
            if not expanded or len(expanded) < len(query):
                expanded = query
            
            logger.debug(f"Query expansion: '{query}' -> '{expanded}'")
            return expanded
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
    
    def _generate_search_keywords(self, query: str, technical_terms: List[str]) -> List[str]:
        """Generate optimized search keywords"""
        keywords = []
        
        # Extract important words from original query
        import re
        words = re.findall(r'\b\w{3,}\b', query.lower())
        keywords.extend(words)
        
        # Add technical terms
        keywords.extend(technical_terms)
        
        # Add NVMe-specific context keywords based on query content
        nvme_keywords = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['queue', 'submission', 'completion']):
            nvme_keywords.extend(['queue', 'submission', 'completion', 'doorbell'])
        
        if any(word in query_lower for word in ['command', 'cmd']):
            nvme_keywords.extend(['command', 'opcode', 'execution'])
        
        if any(word in query_lower for word in ['memory', 'address', 'dma']):
            nvme_keywords.extend(['memory', 'address', 'dma', 'prp', 'sgl'])
        
        if any(word in query_lower for word in ['namespace', 'ns']):
            nvme_keywords.extend(['namespace', 'nsid', 'format'])
        
        keywords.extend(nvme_keywords)
        
        # Remove duplicates and return unique keywords
        return list(set(keywords))
    
    def _calculate_confidence(self, query: str, technical_terms: List[str]) -> float:
        """Calculate confidence score for the query analysis"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on technical terms found
        if technical_terms:
            confidence += min(0.3, len(technical_terms) * 0.1)
        
        # Increase confidence for specific question patterns
        query_lower = query.lower()
        if any(word in query_lower for word in ['what', 'how', 'why', 'when', 'where']):
            confidence += 0.1
        
        # Increase confidence for NVMe-specific terms
        nvme_specific = ['nvme', 'submission queue', 'completion queue', 'pcie', 'controller']
        if any(term in query_lower for term in nvme_specific):
            confidence += 0.1
        
        # Ensure confidence is between 0 and 1
        return min(1.0, max(0.0, confidence))
    
    def translate_for_retrieval(self, query: str, chat_history: List[ChatMessage] = None) -> str:
        """
        Main method to translate query for optimal retrieval
        
        Args:
            query: Original user query
            chat_history: Previous conversation context
            
        Returns:
            Optimized query for retrieval
        """
        analysis = self.analyze_query(query, chat_history)
        
        # Use expanded query if confidence is high enough
        if analysis.confidence > 0.6:
            return analysis.expanded_query
        else:
            # For low confidence, use a combination of original and keywords
            keywords_str = " ".join(analysis.search_keywords[:5])  # Top 5 keywords
            return f"{query} {keywords_str}"