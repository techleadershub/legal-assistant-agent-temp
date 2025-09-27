"""
Phase 1: Basic LangGraph ReAct Legal Assistant
Simple implementation for hackathon - 3.5 hours to wow!

Core features:
1. LangGraph StateGraph with ReAct pattern
2. 3 basic tools (search, explain, greet)
3. Visible reasoning process
4. Simple Streamlit UI
"""

import streamlit as st
import os
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, TypedDict
import json

# Core dependencies
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.docstore.document import Document
    from langgraph.graph import StateGraph, END
    from langchain.tools import BaseTool
    from pydantic import BaseModel, Field
    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import pypdf
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.info("Run: pip install langchain-google-genai langgraph faiss-cpu sentence-transformers pypdf")
    st.stop()

# Page config
st.set_page_config(page_title="ReAct Legal Assistant", page_icon="⚖️", layout="wide")

# Agent State
class AgentState(TypedDict):
    """State for the ReAct agent."""
    messages: List[Any]
    user_query: str
    reasoning: str
    action: str
    tool_result: str
    final_response: str
    # Phase 3: Multi-step reasoning
    step_count: int
    reasoning_chain: List[Dict[str, str]]
    context_memory: str
    is_complete: bool

# Simplified Tool Definitions (avoiding Pydantic compatibility issues)
class SearchDocumentsTool:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Search documents for relevant content."""
        if not self.vector_store or not self.vector_store.documents:
            return "No documents loaded. Please upload a document first."
        
        # Simple search
        results = self.vector_store.search(query, num_results=3)
        return results

class ExplainContentTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Explain content in simple terms."""
        prompt = f"""Explain this legal content in simple terms:

{content}

Provide a clear, easy-to-understand explanation."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error explaining content: {e}"

class GreetUserTool:
    def _run(self, query: str) -> str:
        """Greet user and provide help."""
        return """Hello! I'm your Advanced ReAct Legal Assistant. I can help you with legal documents by:

🔍 **Searching** for specific clauses and sections
📖 **Explaining** legal terms in simple language
⚖️ **Analyzing** contract terms and risks
📋 **Contract Analysis** - Risk assessment and clause analysis
💰 **Cost Analysis** - Financial impact calculations
📊 **Compliance Checking** - Regulatory compliance verification
🔍 **Clause Finding** - Specific legal clause identification
📝 **Summary Generation** - Executive summaries
⚠️ **Risk Assessment** - Legal risk identification

Upload a document and ask me questions like:
- "Analyze the risks in this contract"
- "Check compliance with employment law"
- "Calculate the financial impact of termination"
- "Find all liability clauses"
- "Generate an executive summary"

I use advanced reasoning to understand your questions and provide the best answers!"""

# Phase 2: Legal-Specific Tools
class ContractAnalyzerTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Analyze contract for risks and key terms."""
        prompt = f"""Analyze this legal contract and provide a comprehensive risk assessment:

{content}

Please provide:
1. **Key Risks** - Major potential issues
2. **Critical Clauses** - Important terms to review
3. **Financial Impact** - Cost implications
4. **Compliance Issues** - Regulatory concerns
5. **Recommendations** - Suggested actions

Format as a professional legal analysis."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error analyzing contract: {e}"

class ComplianceCheckerTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Check contract compliance with regulations."""
        prompt = f"""Review this contract for compliance with employment law, data protection, and other relevant regulations:

{content}

Please check for:
1. **Employment Law Compliance** - Labor standards, discrimination
2. **Data Protection** - GDPR, privacy requirements
3. **Industry Regulations** - Sector-specific rules
4. **Contract Law** - Standard legal requirements
5. **Risk Areas** - Potential compliance issues

Provide specific compliance recommendations."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error checking compliance: {e}"

class CostCalculatorTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Calculate financial impact of contract terms."""
        prompt = f"""Analyze the financial implications of this contract:

{content}

Please calculate:
1. **Termination Costs** - Severance, notice periods
2. **Penalty Clauses** - Financial penalties
3. **Payment Terms** - Cash flow implications
4. **Insurance Costs** - Liability coverage
5. **Total Financial Impact** - Overall cost analysis

Provide specific financial calculations and recommendations."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error calculating costs: {e}"

class ClauseFinderTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Find specific legal clauses in contract."""
        prompt = f"""Identify and extract specific legal clauses from this contract:

{content}

Please find and explain:
1. **Termination Clauses** - How to end the contract
2. **Liability Clauses** - Responsibility and damages
3. **Payment Clauses** - Financial obligations
4. **Confidentiality Clauses** - Information protection
5. **Dispute Resolution** - How conflicts are handled

Provide the exact text and explanations for each clause."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error finding clauses: {e}"

class SummaryGeneratorTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Generate executive summary of contract."""
        prompt = f"""Create a professional executive summary of this contract:

{content}

Please provide:
1. **Contract Overview** - What it covers
2. **Key Terms** - Most important provisions
3. **Financial Summary** - Costs and payments
4. **Risk Assessment** - Major concerns
5. **Action Items** - Next steps required

Format as a professional executive summary."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error generating summary: {e}"

class RiskAssessorTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Assess legal risks in contract."""
        prompt = f"""Conduct a comprehensive risk assessment of this contract:

{content}

Please evaluate:
1. **Legal Risks** - Potential legal issues
2. **Financial Risks** - Cost implications
3. **Operational Risks** - Business impact
4. **Compliance Risks** - Regulatory concerns
5. **Risk Mitigation** - How to reduce risks

Provide a risk score (1-10) and specific recommendations."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error assessing risks: {e}"

# Phase 3: Advanced Tools for Multi-Step Reasoning
class DeepSearchTool:
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def _run(self, query: str) -> str:
        """Advanced search with multiple query strategies."""
        # Try multiple search strategies
        search_queries = [query]
        
        # Add semantic variations
        if "risk" in query.lower():
            search_queries.extend(["liability", "damages", "responsibility", "penalty"])
        elif "compliance" in query.lower():
            search_queries.extend(["regulation", "law", "requirement", "standard"])
        elif "termination" in query.lower():
            search_queries.extend(["end", "terminate", "notice", "severance"])
        
        results = []
        for search_query in search_queries:
            result = self.vector_store.search(search_query, num_results=2)
            if "No relevant sections found" not in result:
                results.append(f"**Search: {search_query}**\n{result}\n")
        
        return "\n".join(results) if results else "No relevant information found."

class RiskAnalyzerTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Advanced risk analysis with scoring."""
        prompt = f"""Perform a comprehensive risk analysis of this contract content:

{content}

Provide:
1. **Risk Score** (1-10 scale)
2. **High-Risk Areas** - Critical issues
3. **Medium-Risk Areas** - Moderate concerns  
4. **Low-Risk Areas** - Minor issues
5. **Risk Mitigation** - Specific recommendations
6. **Overall Assessment** - Summary and next steps

Format as a professional risk assessment report."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error in risk analysis: {e}"

class ComplianceCheckerTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Check regulatory compliance."""
        prompt = f"""Analyze this contract for regulatory compliance:

{content}

Check for:
1. **Employment Law** - Labor standards, discrimination
2. **Data Protection** - Privacy, GDPR compliance
3. **Industry Regulations** - Sector-specific rules
4. **Contract Law** - Standard legal requirements
5. **Compliance Score** - Overall compliance rating

Provide specific compliance recommendations and action items."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error checking compliance: {e}"

class DecisionMakerTool:
    def __init__(self, llm):
        self.llm = llm
    
    def _run(self, content: str) -> str:
        """Make final decision based on analysis."""
        prompt = f"""Based on the comprehensive analysis, provide a final decision:

{content}

Provide:
1. **Final Recommendation** - Accept, reject, or negotiate
2. **Key Concerns** - Main issues to address
3. **Negotiation Points** - Terms to discuss
4. **Next Steps** - Specific actions to take
5. **Confidence Level** - How certain is this recommendation

Format as a professional legal recommendation."""
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error in decision making: {e}"

# Simple Vector Store
class SimpleVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.tfidf_matrix = None
        self.documents = []
        self.document_texts = []
    
    def add_documents(self, documents):
        """Add documents to vector store."""
        self.documents = documents
        self.document_texts = [doc.page_content for doc in documents]
        
        # Create TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(self.document_texts)
    
    def search(self, query, num_results=3):
        """Search for relevant documents."""
        print(f"DEBUG: Search called with query: '{query}'")
        print(f"DEBUG: Vector store has {len(self.documents) if self.documents else 0} documents")
        print(f"DEBUG: TF-IDF matrix is {'None' if self.tfidf_matrix is None else 'available'}")
        
        if self.tfidf_matrix is None or not self.documents:
            print("DEBUG: No documents available for search")
            return "No documents available for search."
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        print(f"DEBUG: Similarities: {similarities}")
        
        # Get top results
        top_indices = similarities.argsort()[-num_results:][::-1]
        print(f"DEBUG: Top indices: {top_indices}")
        
        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # Only include results with some similarity
                doc = self.documents[idx]
                results.append(f"**Section {i+1}** (Relevance: {similarities[idx]:.2f})\n{doc.page_content[:300]}...\n")
        
        print(f"DEBUG: Found {len(results)} results")
        return "\n".join(results) if results else "No relevant sections found."

# ReAct Legal Agent
class ReActLegalAgent:
    """LangGraph ReAct Legal Assistant Agent."""
    
    def __init__(self, api_key: str):
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            google_api_key=api_key,
            temperature=0.3
        )
        
        # Initialize vector store
        self.vector_store = SimpleVectorStore()
        
        # Initialize tools (Phase 3: Advanced multi-step reasoning)
        self.tools = [
            SearchDocumentsTool(self.vector_store),
            ExplainContentTool(self.llm),
            GreetUserTool(),
            ContractAnalyzerTool(self.llm),
            SummaryGeneratorTool(self.llm),
            # Phase 3: Advanced tools
            DeepSearchTool(self.vector_store),
            RiskAnalyzerTool(self.llm),
            ComplianceCheckerTool(self.llm),
            DecisionMakerTool(self.llm)
        ]
        
        # Tools are available for direct use
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow with multi-step reasoning loop."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("reasoning", self._reasoning_node)
        workflow.add_node("action", self._action_node)
        workflow.add_node("response", self._response_node)
        
        # Add edges with entry point
        workflow.set_entry_point("reasoning")
        workflow.add_edge("reasoning", "action")
        workflow.add_edge("action", "response")
        
        # Add conditional edge for multi-step reasoning
        def should_continue(state: AgentState) -> str:
            """Decide whether to continue multi-step reasoning or end."""
            is_complete = state.get("is_complete", False)
            step_count = state.get("step_count", 0)
            
            # Continue if not complete and under 3 steps
            if not is_complete and step_count < 3:
                return "reasoning"  # Loop back to reasoning for next step
            else:
                return "end"  # End the workflow
        
        workflow.add_conditional_edges(
            "response",
            should_continue,
            {
                "reasoning": "reasoning",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _reasoning_node(self, state: AgentState) -> AgentState:
        """Reasoning node - decides what action to take with multi-step reasoning."""
        user_query = state["user_query"]
        step_count = state.get("step_count", 0)
        context_memory = state.get("context_memory", "")
        reasoning_chain = state.get("reasoning_chain", [])
        
        # Multi-step reasoning logic
        if step_count == 0:
            # First step - analyze the query and plan the approach
            reasoning_prompt = f"""You are an Advanced Legal Assistant Agent. You MUST use multi-step reasoning for complex queries.

User Query: "{user_query}"

Available tools:
1. search_documents - Search for relevant content
2. explain_content - Explain legal content in simple terms  
3. greet_user - Greet user and provide help
4. contract_analyzer - Analyze contract for risks and key terms
5. summary_generator - Generate executive summary
6. deep_search - Advanced search with multiple strategies
7. risk_analyzer - Comprehensive risk analysis
8. compliance_checker - Check regulatory compliance
9. decision_maker - Make final recommendation

IMPORTANT: For complex queries like "Is this contract risky?", "Should I sign?", "What are the risks?", you MUST use multi-step reasoning:

Step 1: ALWAYS start with deep_search to gather comprehensive information
Step 2: Then use risk_analyzer to analyze the findings
Step 3: Finally use decision_maker to make the recommendation

For the query "{user_query}", you MUST start with deep_search to gather information first.

Respond with:
REASONING: [Your reasoning and plan for multi-step approach]
ACTION: [EXACT tool name - MUST be deep_search for complex queries]"""
        
        elif step_count < 3:
            # Middle steps - continue based on previous results
            previous_results = "\n".join([f"Step {i+1}: {step['action']} - {step['result'][:200]}..." 
                                       for i, step in enumerate(reasoning_chain)])
            
            reasoning_prompt = f"""You are continuing a multi-step legal analysis. You MUST continue the multi-step process.

User Query: "{user_query}"
Previous Context: {context_memory}
Previous Steps: {previous_results}

Available tools:
1. search_documents - Search for relevant content
2. explain_content - Explain legal content in simple terms  
3. contract_analyzer - Analyze contract for risks and key terms
4. summary_generator - Generate executive summary
5. deep_search - Advanced search with multiple strategies
6. risk_analyzer - Comprehensive risk analysis
7. compliance_checker - Check regulatory compliance
8. decision_maker - Make final recommendation

MULTI-STEP CONTINUATION RULES:
- If you just did deep_search (Step 1), next use risk_analyzer (Step 2)
- If you just did risk_analyzer (Step 2), next use decision_maker (Step 3)
- If you just did compliance_checker, next use decision_maker
- NEVER skip steps or go backwards

Continue the analysis with the next logical step.

Respond with:
REASONING: [Your reasoning for next step in the multi-step process]
ACTION: [EXACT tool name - follow the step sequence]"""
        
        else:
            # Final step - make decision
            reasoning_prompt = f"""You are at the final step of a multi-step legal analysis. Make the final recommendation.

User Query: "{user_query}"
Context: {context_memory}

Use decision_maker to provide final recommendation.

Respond with:
REASONING: [Your reasoning for final step]
ACTION: decision_maker"""

        try:
            response = self.llm.invoke(reasoning_prompt)
            content = response.content
            
            # Extract reasoning and action
            reasoning = content.split("REASONING:")[1].split("ACTION:")[0].strip() if "REASONING:" in content else "Analyzing query..."
            action = content.split("ACTION:")[1].strip() if "ACTION:" in content else "greet_user"
            
            # Clean up action name and ensure it's valid
            action = action.strip().lower()
            valid_actions = ["search_documents", "explain_content", "greet_user", "contract_analyzer", "summary_generator", "deep_search", "risk_analyzer", "compliance_checker", "decision_maker"]
            if action not in valid_actions:
                # Try to find a partial match
                for valid_action in valid_actions:
                    if valid_action in action or action in valid_action:
                        action = valid_action
                        break
                else:
                    action = "greet_user"  # Default fallback
            
            state["reasoning"] = reasoning
            state["action"] = action
            
        except Exception as e:
            state["reasoning"] = f"Error in reasoning: {e}"
            state["action"] = "greet_user"
        
        return state
    
    def _action_node(self, state: AgentState) -> AgentState:
        """Action node - executes the selected tool."""
        action = state["action"]
        user_query = state["user_query"]
        
        # Debug: Print the action being executed
        print(f"DEBUG: Executing action: '{action}'")
        
        try:
            # Execute the tool based on action
            if action == "search_documents":
                # For search, try multiple query variations to find relevant content
                search_queries = [user_query]
                
                # Add specific variations for common legal queries
                if "termination" in user_query.lower():
                    search_queries.extend(["termination", "end", "terminate", "clause"])
                elif "payment" in user_query.lower():
                    search_queries.extend(["payment", "pay", "salary", "compensation"])
                elif "liability" in user_query.lower():
                    search_queries.extend(["liability", "responsibility", "damages"])
                
                result = ""
                for query in search_queries:
                    result = self.tools[0]._run(query)
                    if "No relevant sections found" not in result:
                        break
            elif action == "explain_content":
                # For explanation, use the original query but also try common legal terms
                search_queries = [user_query, "termination", "clause", "agreement"]
                search_result = ""
                for query in search_queries:
                    search_result = self.tools[0]._run(query)
                    if "No relevant sections found" not in search_result:
                        break
                
                if "No documents loaded" in search_result or "No relevant sections found" in search_result:
                    result = "I couldn't find relevant information. Please make sure you've uploaded a document."
                else:
                    result = self.tools[1]._run(search_result)
            elif action == "contract_analyzer":
                # For contract analysis, search for general contract content
                search_result = self.tools[0]._run("contract terms agreement")
                if "No documents loaded" in search_result or "No relevant sections found" in search_result:
                    result = "I couldn't find contract information. Please upload a legal document first."
                else:
                    result = self.tools[3]._run(search_result)
            elif action == "summary_generator" or "summary" in action.lower():
                # For summary generation, search for general contract content
                search_result = self.tools[0]._run("contract terms agreement")
                if "No documents loaded" in search_result or "No relevant sections found" in search_result:
                    result = "I couldn't find contract information. Please upload a legal document first."
                else:
                    result = self.tools[4]._run(search_result)
            elif action == "deep_search":
                # Phase 3: Advanced search
                result = self.tools[5]._run(user_query)
            elif action == "risk_analyzer":
                # Phase 3: Risk analysis
                search_result = self.tools[0]._run("liability risk terms")
                if "No documents loaded" in search_result or "No relevant sections found" in search_result:
                    result = "I couldn't find contract information. Please upload a legal document first."
                else:
                    result = self.tools[6]._run(search_result)
            elif action == "compliance_checker":
                # Phase 3: Compliance check
                search_result = self.tools[0]._run("compliance regulation law")
                if "No documents loaded" in search_result or "No relevant sections found" in search_result:
                    result = "I couldn't find contract information. Please upload a legal document first."
                else:
                    result = self.tools[7]._run(search_result)
            elif action == "decision_maker":
                # Phase 3: Final decision
                result = self.tools[8]._run(state.get("context_memory", ""))
            else:
                result = self.tools[2]._run(user_query)  # greet_user
            
            state["tool_result"] = result

        except Exception as e:
            state["tool_result"] = f"Error executing action: {e}"
        
        return state
    
    def _response_node(self, state: AgentState) -> AgentState:
        """Response node - handles multi-step reasoning and generates response."""
        user_query = state["user_query"]
        tool_result = state["tool_result"]
        action = state["action"]
        step_count = state.get("step_count", 0)
        reasoning_chain = state.get("reasoning_chain", [])
        context_memory = state.get("context_memory", "")
        
        # Multi-step reasoning logic
        if step_count == 0:
            # First step - initialize reasoning chain
            reasoning_chain = [{
                "step": 1,
                "action": action,
                "reasoning": state["reasoning"],
                "result": tool_result
            }]
            context_memory = tool_result
            step_count = 1
            is_complete = False
            
        elif step_count < 3 and action != "decision_maker":
            # Continue multi-step reasoning
            reasoning_chain.append({
                "step": step_count + 1,
                "action": action,
                "reasoning": state["reasoning"],
                "result": tool_result
            })
            context_memory += f"\n\nStep {step_count + 1} Results:\n{tool_result}"
            step_count += 1
            is_complete = False
            
        else:
            # Final step - complete the analysis
            reasoning_chain.append({
                "step": step_count + 1,
                "action": action,
                "reasoning": state["reasoning"],
                "result": tool_result
            })
            is_complete = True
        
        # Real-time console logging
        print(f"📋 STEP {step_count}: Executing {action}")
        print(f"💭 REASONING: {state['reasoning']}")
        print(f"🔄 CHAIN LENGTH: {len(reasoning_chain)} steps")
        
        # Update state
        state["step_count"] = step_count
        state["reasoning_chain"] = reasoning_chain
        state["context_memory"] = context_memory
        state["is_complete"] = is_complete
        
        # Generate response based on completion status
        if is_complete:
            final_response = tool_result
        elif "No documents loaded" in tool_result:
            final_response = "Please upload a document first, then ask me questions about specific clauses or terms."
        elif "No relevant sections found" in tool_result:
            final_response = "I couldn't find relevant information for your query. Try asking about different topics or make sure you've uploaded a document."
        else:
            # Multi-step in progress
            final_response = f"**Step {step_count} Complete**\n\n{tool_result}\n\n*Continuing analysis...*"
        
        state["final_response"] = final_response
        return state
    
    def process_document(self, file_content: bytes, filename: str) -> bool:
        """Process PDF document and create searchable chunks."""
        try:
            # Extract text from PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            os.unlink(tmp_file_path)
            
            # Debug: Print extracted text length
            print(f"DEBUG: Extracted text length: {len(text)} characters")
            
            # Split into chunks
            chunks = self._split_text(text, filename)
            print(f"DEBUG: Created {len(chunks)} chunks")
            
            if not chunks:
                print("DEBUG: No chunks created!")
                return False
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            print(f"DEBUG: Vector store now has {len(self.vector_store.documents)} documents")
            return True

        except Exception as e:
            st.error(f"Error processing document: {e}")
            return False
    
    def _split_text(self, text: str, source: str) -> List[Document]:
        """Split text into chunks for vector search."""
        words = text.split()
        chunks = []
        chunk_size = 1000
        
        for i in range(0, len(words), chunk_size - 200):
            chunk_text = " ".join(words[i:i + chunk_size])
            if chunk_text.strip():
                chunks.append(Document(
                    page_content=chunk_text,
                    metadata={'source': source, 'chunk_id': len(chunks)}
                ))
        
        return chunks
    
    def process_query(self, user_query: str) -> Dict[str, str]:
        """Process a user query using multi-step ReAct pattern."""
        # Initialize state for multi-step reasoning
        initial_state = {
            "messages": [],
            "user_query": user_query,
            "reasoning": "",
            "action": "",
            "tool_result": "",
            "final_response": "",
            "step_count": 0,
            "reasoning_chain": [],
            "context_memory": "",
            "is_complete": False
        }
        
        try:
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return {
                "reasoning": final_state["reasoning"],
                "action": final_state["action"],
                "tool_result": final_state["tool_result"],
                "final_response": final_state["final_response"],
                "step_count": final_state.get("step_count", 0),
                "reasoning_chain": final_state.get("reasoning_chain", []),
                "is_complete": final_state.get("is_complete", False)
            }
            
        except Exception as e:
            return {
                "reasoning": f"Error in reasoning: {e}",
                "action": "error",
                "tool_result": f"Error: {e}",
                "final_response": f"I encountered an error: {e}",
                "step_count": 0,
                "reasoning_chain": [],
                "is_complete": True
            }

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("⚖️ Advanced ReAct Legal Assistant Agent")
    st.markdown("**Phase 3: Multi-Step Reasoning with LangGraph ReAct Pattern**")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        # API Key check
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ Google API Key not found")
            st.info("Set GOOGLE_API_KEY environment variable")
            st.markdown("Get your free API key at: https://makersuite.google.com/app/apikey")
        else:
            st.success("✅ API Key configured")
            if st.session_state.agent is None:
                st.session_state.agent = ReActLegalAgent(api_key)
        
        # Document upload
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and st.button("📤 Process Document"):
            with st.spinner("Processing document..."):
                success = st.session_state.agent.process_document(
                    uploaded_file.getvalue(),
                    uploaded_file.name
                )
                if success:
                    st.success("Document processed successfully!")
                    st.session_state.conversation = []
                else:
                    st.error("Failed to process document")
        
        # Show ReAct process with enhanced visualization
        if st.session_state.agent:
            st.markdown("---")
            st.markdown("### 🧠 ReAct Process")
            st.markdown("""
            **🧠 Reasoning**: Agent analyzes the query and plans approach
            **🔧 Action**: Agent selects and executes appropriate tools
            **📋 Response**: Agent generates comprehensive answer
            """)
            
            # Show current status
            if st.session_state.conversation:
                last_turn = st.session_state.conversation[-1]
                st.markdown("### 🔄 Current Status")
                st.markdown(f"**Last Action:** `{last_turn['action']}`")
                st.markdown(f"**Last Reasoning:** {last_turn['reasoning'][:100]}...")
    
    # Main chat interface
    st.header("💬 Chat with ReAct Agent")
    
    # Display conversation with enhanced visual thought process
    for turn in st.session_state.conversation:
        st.markdown(f"**You:** {turn['user']}")
        
        # Simple agent response - no steps display
        st.markdown(f"**🤖 Agent:** {turn['action']}")
        
        # Final response with enhanced formatting
        st.markdown("---")
        st.markdown(f"**🤖 Agent Response:**")
        st.markdown(turn['agent'])
        st.markdown("---")
    
    # Chat input (moved completely outside sidebar)
    user_query = st.chat_input("Ask your ReAct legal assistant...")
    
    if user_query and st.session_state.agent:
        # Simple thinking indicator with real-time logging
        with st.spinner("🤔 Agent is thinking..."):
            print(f"\n🚀 STARTING MULTI-STEP ANALYSIS FOR: '{user_query}'")
            # Process query with ReAct
            result = st.session_state.agent.process_query(user_query)
            print(f"✅ ANALYSIS COMPLETE - {len(result.get('reasoning_chain', []))} steps executed")
        
        # Add to conversation with multi-step reasoning
        st.session_state.conversation.append({
            'user': user_query,
            'reasoning': result['reasoning'],
            'action': result['action'],
            'tool_result': result['tool_result'],
            'agent': result['final_response'],
            'step_count': result.get('step_count', 0),
            'reasoning_chain': result.get('reasoning_chain', []),
            'is_complete': result.get('is_complete', False)
        })
        
        st.rerun()
    
    # Example questions section
    st.markdown("---")
    st.header("🎯 Example Questions")
    
    if st.session_state.agent and st.session_state.agent.vector_store.documents:
        examples = [
            "Is this contract risky for me?",
            "Is this employment contract compliant?",
            "Should I sign this contract?",
            "What are the main risks in this agreement?",
            "Analyze the termination terms"
        ]
        
        for example in examples:
            if st.button(example, key=f"example_{hash(example)}"):
                with st.spinner("Processing..."):
                    result = st.session_state.agent.process_query(example)
                    st.session_state.conversation.append({
                        'user': example,
                        'reasoning': result['reasoning'],
                        'action': result['action'],
                        'tool_result': result['tool_result'],
                        'agent': result['final_response']
                    })
                    st.rerun()
    else:
        st.info("Upload a document to see example questions!")
        
        # ReAct explanation with visual flow
        st.markdown("---")
        st.markdown("### 🎓 Advanced ReAct Pattern")
        
        # Visual flow diagram
        st.markdown("""
        ```
        🧠 REASONING → 🔧 ACTION → 📋 RESPONSE
              ↓              ↓           ↓
        Analyze Query → Execute Tool → Generate Answer
              ↓              ↓           ↓
        Plan Approach → Get Results → Final Response
        ```""")
        
        st.markdown("### 🔄 Multi-Step Reasoning Flow")
        st.markdown("""
        ```
        Step 1: 🔍 Search → Gather Information
              ↓
        Step 2: ⚠️ Analyze → Process Findings  
              ↓
        Step 3: 🎯 Decide → Make Recommendation
        ```""")
        
        st.markdown("### 🛠️ Phase 3 Advanced Tools")
        st.markdown("""
        - 🔍 **Search Documents** - Find relevant content
        - 📖 **Explain Content** - Simplify legal terms
        - 👋 **Greet User** - Welcome and help
        - 📋 **Contract Analyzer** - Simple contract analysis
        - 📝 **Summary Generator** - Generate summaries
        - 🔎 **Deep Search** - Advanced multi-strategy search
        - ⚠️ **Risk Analyzer** - Comprehensive risk assessment
        - ⚖️ **Compliance Checker** - Regulatory compliance verification
        - 🎯 **Decision Maker** - Final legal recommendations
        """)

if __name__ == "__main__":
    main()