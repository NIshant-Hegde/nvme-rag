#!/usr/bin/env python3
"""
    Full RAG Demo: Complete Question-Answering (QA) System for NVMe RAG

    This demo showcases the complete QA pipeline including:
    - Query translation and analysis
    - Context retrieval
    - Answer generation
    - Follow-up question suggestions
    - Session management
"""

# imports
import json
import logging
from pathlib import Path
from typing import Dict, Any
from src.pipeline.integration import RAGPipelineIntegration
from src.vector_store.embedding_generator import EmbeddingConfig
from src.llm.ollama_client import OllamaConfig
from src.llm.answer_generator import AnswerGenerationConfig, AnswerStyle
from src.retrieval.retrieval_pipeline import RetrievalConfig, RetrievalStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NVMeQADemo:
    """
        Complete NVMe Question-Answering Demo
    """
    
    def __init__(self):
        """
            Initialize the QA system
        """
        
        # Configuration
        self.vector_store_path = "data/vector_store"
        
        # Component configurations
        self.embedding_config = EmbeddingConfig(
            model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",  #TODO: experiment with different models
            device="cpu"
        )
        
        self.ollama_config = OllamaConfig(
            base_url="http://localhost:11434",
            model="gemma3:12b-it-qat",  # TODO: experiment with different models
            temperature=0.1,
            max_tokens=2048
        )
        
        self.retrieval_config = RetrievalConfig(
            strategy=RetrievalStrategy.RERANKED,  # TODO: experiment with different retrieval strategies
            top_k=5,
            enable_query_enhancement=True,
            enable_context_filtering=True,
            max_context_length=3000
        )
        
        self.answer_config = AnswerGenerationConfig(
            style=AnswerStyle.DETAILED,
            max_answer_length=1000,
            include_sources=True,
            include_confidence=True,
            cite_sections=True,
            explain_technical_terms=False
        )
        
        # Initialize pipeline
        print("\nInitializing NVMe Base Spec QA System...")
        self.rag_pipeline = RAGPipelineIntegration(
            vector_store_path=self.vector_store_path,
            embedding_config=self.embedding_config,
            ollama_config=self.ollama_config,
            retrieval_config=self.retrieval_config,
            answer_config=self.answer_config
        )
        
        # Check system status
        self._check_system_status()
        
        print("\nNVMe Base Spec QA System ready!")
    
    def _check_system_status(self):
        """
        Check and display system status
        """
        try:
            status = self.rag_pipeline.get_pipeline_status()   #status check: fails when ollama client, for example, is not running
            
            print("\nSystem Status:")
            print(f"Overall Status: {status.get('overall_status', 'unknown')}")
            
            components = status.get('components', {})
            for component, info in components.items():
                comp_status = info.get('status', 'unknown')
                print(f"{component}: {comp_status}")
            
            # Vector store info
            vector_info = components.get('vector_store', {})
            chunk_count = vector_info.get('total_chunks', 0)
            if chunk_count > 0:
                print(f"Indexed chunks: {chunk_count}")
            else:
                print("No indexed documents found. Please run document processing first.")
            
        except Exception as e:
            print(f"System status check failed: {e}")
    
    def run_interactive_demo(self):
        """Run interactive question-answering session"""
        
        session_id = None
        
        while True:
            try:
                print("\n" + "-"*150)
                print("Ask questions about NVMe base specification!")
                print("\n                       OR")
                print("\nType 'config' to see current configuration, 'status' to check system status, 'demo' for sample questions, 'quit' to exit, 'help' for commands")
                print("-"*150)
               
                # Get user input
                user_input = input("\nYour question or command: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'demo':
                    self._run_sample_questions()
                    continue
                elif user_input.lower().startswith('config'):
                    self._show_config()
                    continue
                elif user_input.lower().startswith('status'):
                    self._check_system_status()
                    continue
                
                # Process question
                print("\nProcessing...")
                qa_result = self.rag_pipeline.ask_question(
                    question=user_input,
                    session_id=session_id
                )
                
                # Use session from result
                if session_id is None:
                    session_id = qa_result.session_id
                
                # Display results
                self._display_qa_result(qa_result)
                
                '''Commenting out follow-up for now'''
                # Offer follow-up
                #self._handle_follow_up(qa_result)
                
            except KeyboardInterrupt:
                print("\n\nDemo interrupted by user")
                break
            except Exception as e:
                print(f"\nError processing question: {e}")
                logger.error(f"Demo error: {e}", exc_info=True)
        
        print("\nThanks for using the NVMe QA System!")
        
        # Show session summary if we had a session
        if session_id:
            self._show_session_summary(session_id)
    
    def _display_qa_result(self, qa_result):
        """
        Display comprehensive QA result
        """
        print("\n" + "="*150)
        print("ANSWER:")
        print("="*150)
        print(qa_result.generated_answer.answer)
        
        # Show confidence and sources
        confidence = qa_result.generated_answer.confidence
        print(f"\nConfidence: {confidence:.1%}")
        
        # Show content source percentages
        chunk_percentage = qa_result.generated_answer.chunk_content_percentage
        llm_percentage = qa_result.generated_answer.llm_generated_percentage
        #print(f"  Content Sources: {chunk_percentage:.1f}% from retrieved chunks, {llm_percentage:.1f}% LLM generated")
        
        # printing what sources were used to answer the question
        if qa_result.generated_answer.sources:
            print(f"Sources: {len(qa_result.generated_answer.sources)} sections")
            for i, source in enumerate(qa_result.generated_answer.sources[:3], 1):
                section = source.get('section', 'Unknown')
                page = source.get('page', 'N/A')
                print(f"  [{i}] {section} (page {page})")
        
        # Show query analysis
        analysis = qa_result.query_analysis
        print(f"\nQuery Analysis:")
        print(f"Type: {analysis.query_type.value}")
        print(f"Intent: {analysis.intent}")
        if analysis.technical_terms:
            print(f"Technical terms: {', '.join(analysis.technical_terms[:5])}")
        
        # Show processing stats
        print(f"\nProcessing: {qa_result.processing_time_seconds:.2f}s")
        print(f"Context chunks: {qa_result.generated_answer.context_used}")
        
        # disabling this for now
        '''
        # Show follow-up questions
        if qa_result.follow_up_questions:
            print("\n  Suggested follow-up questions:")
            for i, question in enumerate(qa_result.follow_up_questions, 1):
                print(f"  {i}. {question}")
        '''
    
    def _handle_follow_up(self, qa_result):
        """
        Handle follow-up question selection
        """
        if not qa_result.follow_up_questions:
            return
        
        while True:
            try:
                choice = input("\nAsk a follow-up? (enter number, 'n' for no, or type your own): ").strip()
                
                if choice.lower() in ['n', 'no', '']:
                    break
                
                # Check if it's a number (selecting from suggestions)
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(qa_result.follow_up_questions):
                        follow_up_question = qa_result.follow_up_questions[idx]
                        print(f"\nFollowing up: {follow_up_question}")
                        
                        # Process follow-up
                        follow_up_result = self.rag_pipeline.ask_follow_up(
                            follow_up_question=follow_up_question,
                            previous_result=qa_result
                        )
                        
                        self._display_qa_result(follow_up_result)
                        break
                    else:
                        print("Invalid number. Please try again.")
                        continue
                else:
                    # Custom follow-up question
                    print(f"\nFollowing up: {choice}")
                    follow_up_result = self.rag_pipeline.ask_follow_up(
                        follow_up_question=choice,
                        previous_result=qa_result
                    )
                    
                    self._display_qa_result(follow_up_result)
                    break
                    
            except ValueError:
                print("Invalid input. Please try again.")
            except Exception as e:
                print(f"Error with follow-up: {e}")
                break
    
    def _run_sample_questions(self):
        """Run demonstration with sample questions"""
        sample_questions = [
            "What is the purpose of the NVMe submission queue?",
            "How does DMA work in NVMe controllers?",
            "What are the different types of NVMe commands?",
            "How is memory addressing handled in NVMe?",
            "What is the role of completion queues?"
        ]
        
        print("\n Running sample questions demonstration...")
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n{'='*60}")
            print(f"Sample Question {i}/{len(sample_questions)}")
            print(f"{'='*60}")
            print(f"{question}")
            
            try:
                qa_result = self.rag_pipeline.ask_question(question)
                self._display_qa_result(qa_result)
                
                # Pause between questions
                if i < len(sample_questions):
                    input("\nPress Enter to continue to next question...")
                    
            except Exception as e:
                print(f"Error with sample question: {e}")
        
        print("\nSample demonstration completed!")
    
    def _show_help(self):
        """Show help information"""
        print("\nAvailable Commands:")
        print("help     - Show this help message")
        print("demo     - Run sample questions")
        print("config   - Show current configuration")
        print("status   - Check system status")
        print("quit     - Exit the demo")
        print("\nTips:")
        print(" - Ask specific technical questions about NVMe")
        print(" - Use follow-up questions to dive deeper")
        print(" - The system maintains conversation context")
    
    def _show_config(self):
        """Show current configuration"""
        print("\nCurrent Configuration:")
        print(f"Vector Store: {self.vector_store_path}")
        print(f"LLM Model: {self.ollama_config.model}")
        print(f"Retrieval Strategy: {self.retrieval_config.strategy.value}")
        print(f"Answer Style: {self.answer_config.style.value}")
        print(f"Max Context: {self.retrieval_config.max_context_length}")
        print(f"Top-K Results: {self.retrieval_config.top_k}")
    
    def _show_session_summary(self, session_id: str):
        """Show session summary"""
        try:
            summary = self.rag_pipeline.get_conversation_summary(session_id)
            
            print("\nSession Summary:")
            print(f"Questions Asked: {summary.get('total_exchanges', 0)}")
            print(f"Session Duration: {summary.get('session_duration_minutes', 0):.1f} minutes")
            
            topics = summary.get('topics_discussed', [])
            if topics:
                print(f"Topics Discussed: {', '.join(topics)}")
                
        except Exception as e:
            print(f"Could not generate session summary: {e}")
    
    def export_demo_results(self, qa_result, filename: str = None):
        """Export QA result to JSON file"""
        if filename is None:
            filename = f"qa_result_{qa_result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            exported_data = self.rag_pipeline.export_qa_result(qa_result)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(exported_data, f, indent=2, ensure_ascii=False)
            
            print(f"Results exported to: {filename}")
            
        except Exception as e:
            print(f"Export failed: {e}")


def main():
    """Main demo function"""
    try:
        # Initialize demo
        demo = NVMeQADemo()
        
        # Run interactive demo
        demo.run_interactive_demo()
        
    except KeyboardInterrupt:
        print("\n\nDemo terminated by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)
    finally:
        print("\nCleaning up...")
        try:
            if 'demo' in locals():
                demo.rag_pipeline.cleanup()
        except Exception as e:
            print(f"Cleanup warning: {e}")


if __name__ == "__main__":
    main()