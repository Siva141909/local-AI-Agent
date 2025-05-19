import streamlit as st
import os
import json
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from langsmith import Client
from langsmith import traceable
import langflow
from langflow.graph import Graph
import traceback
import tempfile

# Set API Keys (better to use st.secrets or environment variables)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDBHqDP8frO-os-FeAlvQc2nb5ENKUhCMw"  # Replace with your actual key
os.environ["LANGCHAIN_API_KEY"] = "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"  # Replace with your actual key
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Multi-Agent Editor"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"  # LangSmith endpoint

# Initialize LangSmith client
@st.cache_resource
def get_langsmith_client():
    try:
        return Client()
    except Exception as e:
        st.error(f"Failed to initialize LangSmith client: {str(e)}")
        return None

# Configure Streamlit
st.set_page_config(page_title="Multi-Agent Editor", layout="wide")
st.title("🧠 Integrated Multi-Agent File Editor")
st.caption("Using LangChain, LangGraph, LangSmith and LangFlow")

# Initialize LLM with error handling
@st.cache_resource
def get_llm():
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

llm = get_llm()
langsmith_client = get_langsmith_client()

# UI Components
folder_path = st.sidebar.text_input("📁 Enter directory path:", value=str(Path.cwd()))

def list_files(path):
    try:
        return [f for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.endswith(('.py', '.txt', '.md', '.json', '.csv'))]
    except Exception as e:
        st.error(f"Error accessing directory: {str(e)}")
        return []

# File selection logic
if not os.path.isdir(folder_path):
    st.error("🚫 Invalid directory path.")
else:
    files = list_files(folder_path)

    if not files:
        st.info("📂 No editable files found.")
    else:
        selected_file = st.sidebar.selectbox("Select file to edit", files)
        file_path = os.path.join(folder_path, selected_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            content = ""

        st.markdown(f"### ✏️ Editing: `{selected_file}`")
        new_content = st.text_area("Edit the content below:", value=content, height=400)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Changes"):
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    st.success("✅ Saved!")
                except Exception as e:
                    st.error(f"Error saving file: {str(e)}")

        with col2:
            try:
                with open(file_path, 'rb') as f:
                    st.download_button("⬇️ Download", data=f, file_name=selected_file, mime="text/plain")
            except Exception as e:
                st.error(f"Error preparing download: {str(e)}")

        st.divider()
        st.subheader("🤖 Integrated Multi-Agent Edit System")

        prompt = st.text_area("Enter your editing instruction (e.g., 'Refactor the code to improve error handling'):")
        
        # Show advanced options in a collapsible section
        with st.sidebar.expander("Advanced Options"):
            enable_langflow_vis = st.checkbox("Enable LangFlow Visualization", value=True)
            enable_langsmith_tracing = st.checkbox("Enable LangSmith Tracing", value=True)
            chunk_size = st.slider("Text Chunk Size (for large files)", 
                                  min_value=500, max_value=8000, value=4000, step=500)
            agent_temperature = st.slider("Agent Temperature", 
                                         min_value=0.0, max_value=1.0, value=0.2, step=0.1)

        # Multi-agent system
        run_button = st.button("🚀 Run Multi-Agent Edit")
        
        if run_button and prompt and llm:
            with st.spinner("Agents are working on your request..."):
                try:
                    # Define state schema with proper typing
                    class EditState(TypedDict):
                        instruction: str
                        content: str
                        chunks: Optional[List[str]]
                        draft: str
                        review: str
                        output: str
                        metadata: Dict[str, Any]
                    
                    # Create a text splitter for larger files
                    @traceable
                    def chunk_text(text: str, chunk_size: int = 4000) -> List[str]:
                        """Split text into manageable chunks for processing."""
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", " ", ""]
                        )
                        docs = text_splitter.create_documents([text])
                        return [doc.page_content for doc in docs]
                    
                    # Define preprocessing node to handle large files
                    @traceable
                    def preprocessor(state: EditState) -> Dict[str, Any]:
                        """Preprocesses the input by chunking if necessary and preparing metadata."""
                        # Check if file is large enough to need chunking
                        if len(state['content']) > chunk_size:
                            chunks = chunk_text(state['content'], chunk_size)
                        else:
                            chunks = [state['content']]
                            
                        # Extract file metadata
                        file_ext = os.path.splitext(selected_file)[1]
                        
                        new_state = state.copy()
                        new_state["chunks"] = chunks
                        new_state["metadata"] = {
                            "filename": selected_file,
                            "file_extension": file_ext,
                            "file_size": len(state['content']),
                            "chunk_count": len(chunks)
                        }
                        return new_state
                    
                    # Define agents with improved prompts and state handling
                    @traceable
                    def writer_agent(state: EditState) -> Dict[str, Any]:
                        """Writer agent that creates an initial draft based on instructions."""
                        # Create writer LangChain
                        writer_template = """You are a professional developer tasked with editing code based on instructions.
                        
                        ORIGINAL CODE:
                        ```
                        {content}
                        ```
                        
                        INSTRUCTION: {instruction}
                        
                        FILE METADATA: {metadata}
                        
                        Please provide an improved version of the code based on the instruction.
                        Only return the complete code without explanations.
                        """
                        
                        writer_prompt = PromptTemplate(
                            template=writer_template,
                            input_variables=["content", "instruction", "metadata"]
                        )
                        
                        writer_chain = LLMChain(
                            llm=ChatGoogleGenerativeAI(
                                model="gemini-2.0-flash", 
                                temperature=agent_temperature
                            ),
                            prompt=writer_prompt,
                            output_key="draft"
                        )
                        
                        # Execute chain
                        result = writer_chain.invoke({
                            "content": state["content"],
                            "instruction": state["instruction"],
                            "metadata": json.dumps(state.get("metadata", {}))
                        })
                        
                        # Create new state with all fields to prevent missing keys
                        new_state = state.copy()
                        new_state["draft"] = result["draft"]
                        return new_state
                    
                    @traceable
                    def reviewer_agent(state: EditState) -> Dict[str, Any]:
                        """Reviewer agent that evaluates the draft and provides feedback."""
                        # Create reviewer LangChain with input-output parser
                        review_template = """You are a code reviewer examining a draft code change.
                        
                        ORIGINAL CODE:
                        ```
                        {content}
                        ```
                        
                        DRAFT CODE:
                        ```
                        {draft}
                        ```
                        
                        INSTRUCTION: {instruction}
                        
                        FILE METADATA: {metadata}
                        
                        Please review the draft carefully and provide specific, actionable feedback:
                        1. Identify any issues or bugs
                        2. Suggest improvements for readability and efficiency
                        3. Point out any missing features relative to the instruction
                        
                        Be concise but comprehensive in your review.
                        """
                        
                        review_prompt = PromptTemplate(
                            template=review_template,
                            input_variables=["content", "draft", "instruction", "metadata"]
                        )
                        
                        # Combine components into a chain with parsers
                        review_chain = (
                            {"content": lambda x: x["content"],
                             "draft": lambda x: x["draft"],
                             "instruction": lambda x: x["instruction"],
                             "metadata": lambda x: json.dumps(x.get("metadata", {}))}
                            | review_prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        # Execute chain
                        review_result = review_chain.invoke({
                            "content": state["content"],
                            "draft": state["draft"],
                            "instruction": state["instruction"],
                            "metadata": state.get("metadata", {})
                        })
                        
                        # Create new state with all fields to prevent missing keys
                        new_state = state.copy()
                        new_state["review"] = review_result
                        return new_state
                    
                    @traceable
                    def optimizer_agent(state: EditState) -> Dict[str, Any]:
                        """Optimizer agent that produces the final code based on the review."""
                        optimize_template = """You are an expert code optimizer.
                        
                        DRAFT CODE:
                        ```
                        {draft}
                        ```
                        
                        REVIEW FEEDBACK:
                        ```
                        {review}
                        ```
                        
                        INSTRUCTION: {instruction}
                        
                        FILE METADATA: {metadata}
                        
                        Based on the review feedback, please provide an optimized final version of the code.
                        Return ONLY the complete, improved code without any explanations or comments about what you changed.
                        """
                        
                        optimize_prompt = PromptTemplate(
                            template=optimize_template,
                            input_variables=["draft", "review", "instruction", "metadata"]
                        )
                        
                        optimize_chain = LLMChain(
                            llm=ChatGoogleGenerativeAI(
                                model="gemini-2.0-flash", 
                                temperature=max(0.1, agent_temperature - 0.1)  # Slightly lower temp for final output
                            ),
                            prompt=optimize_prompt,
                            output_key="output"
                        )
                        
                        # Execute chain
                        result = optimize_chain.invoke({
                            "draft": state["draft"],
                            "review": state["review"],
                            "instruction": state["instruction"],
                            "metadata": json.dumps(state.get("metadata", {}))
                        })
                        
                        # Create new state with all fields to prevent missing keys
                        new_state = state.copy()
                        new_state["output"] = result["output"]
                        return new_state
                    
                    # Decision node to determine if more refinement is needed
                    @traceable
                    def evaluator(state: EditState) -> str:
                        """Determines if the output meets requirements or needs another pass."""
                        evaluate_template = """You are an evaluation agent.
                        
                        INSTRUCTION: {instruction}
                        FINAL OUTPUT: 
                        ```
                        {output}
                        ```
                        
                        Does this output fully satisfy the instruction? Answer with either "COMPLETE" if it 
                        satisfies the requirements or "REFINE" if it needs more work.
                        """
                        
                        evaluate_prompt = PromptTemplate(
                            template=evaluate_template,
                            input_variables=["instruction", "output"]
                        )
                        
                        evaluate_chain = (
                            evaluate_prompt 
                            | llm 
                            | StrOutputParser()
                        )
                        
                        result = evaluate_chain.invoke({
                            "instruction": state["instruction"],
                            "output": state["output"]
                        })
                        
                        if "COMPLETE" in result.upper():
                            return "complete"
                        else:
                            return "refine"

                    # Create progress indicators
                    progress_container = st.empty()
                    progress_text = st.empty()
                    
                    # Debug container
                    debug_container = st.expander("Debug Information", expanded=False)
                    
                    # Create progress display function
                    def update_progress(step, percentage):
                        progress_container.progress(percentage)
                        progress_text.text(f"{step}")
                    
                    # Build LangGraph with proper state management
                    builder = StateGraph(EditState)
                    
                    # Add all nodes
                    builder.add_node("preprocessor", preprocessor)
                    builder.add_node("writer", writer_agent)
                    builder.add_node("reviewer", reviewer_agent)
                    builder.add_node("optimizer", optimizer_agent)
                    builder.add_node("evaluator", evaluator)
                    
                    # Set flow
                    builder.set_entry_point("preprocessor")
                    builder.add_edge("preprocessor", "writer")
                    builder.add_edge("writer", "reviewer")
                    builder.add_edge("reviewer", "optimizer")
                    builder.add_edge("optimizer", "evaluator")
                    
                    # Add conditional logic
                    builder.add_conditional_edges(
                        "evaluator",
                        {
                            "complete": END,
                            "refine": "reviewer"  # Loop back to reviewer if we need refinement
                        }
                    )
                    
                    # Compile graph
                    app = builder.compile()
                    
                    # Initialize complete state
                    initial_state = {
                        "instruction": prompt,
                        "content": content,
                        "chunks": None,
                        "draft": "",
                        "review": "",
                        "output": "",
                        "metadata": {}
                    }
                    
                    # Create progress indicators
                    progress_container.empty()
                    progress_text.empty()
                    
                    # Show initial progress
                    update_progress("Starting multi-agent process...", 0)
                    
                    # Create LangFlow visualization if enabled
                    if enable_langflow_vis:
                        try:
                            # Create a LangFlow Graph representation 
                            flow_graph = Graph()
                            
                            # Add nodes to the graph
                            flow_graph.add_node("preprocessor", "Preprocessor")
                            flow_graph.add_node("writer", "Writer Agent")
                            flow_graph.add_node("reviewer", "Reviewer Agent")
                            flow_graph.add_node("optimizer", "Optimizer Agent")
                            flow_graph.add_node("evaluator", "Evaluator")
                            
                            # Add edges to the graph
                            flow_graph.add_edge("preprocessor", "writer")
                            flow_graph.add_edge("writer", "reviewer")
                            flow_graph.add_edge("reviewer", "optimizer")
                            flow_graph.add_edge("optimizer", "evaluator")
                            flow_graph.add_edge("evaluator", "reviewer", condition="refine")
                            flow_graph.add_edge("evaluator", "END", condition="complete")
                            
                            # Generate and save visualization
                            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp:
                                flow_graph.save(tmp.name)
                                st.sidebar.markdown("### 📊 LangFlow Visualization")
                                with open(tmp.name, 'r') as f:
                                    st.sidebar.components.v1.html(f.read(), height=400)
                        except Exception as e:
                            st.sidebar.warning(f"Could not generate LangFlow visualization: {str(e)}")
                    
                    # Run graph with LangSmith tracing if enabled
                    if enable_langsmith_tracing:
                        trace_id = None
                        try:
                            # Start a new trace
                            if langsmith_client:
                                trace = langsmith_client.create_run(
                                    name="Multi-Agent Editor Run",
                                    run_type="chain",
                                    inputs={"instruction": prompt, "file_name": selected_file},
                                    project_name=os.environ.get("LANGCHAIN_PROJECT", "LangGraph Multi-Agent Editor")
                                )
                                trace_id = trace.id
                        except Exception as e:
                            with debug_container:
                                st.warning(f"LangSmith tracing error: {str(e)}")
                    
                    # Step-by-step execution with progress updates
                    update_progress("Preprocessing and analyzing file...", 10)
                    
                    # Run graph
                    result = app.invoke(initial_state)
                    
                    # Update trace with final result if applicable
                    if enable_langsmith_tracing and trace_id and langsmith_client:
                        try:
                            langsmith_client.update_run(
                                trace_id,
                                outputs={"final_output": result.get("output", "")},
                                status="completed"
                            )
                            with debug_container:
                                st.success(f"LangSmith trace completed: {trace_id}")
                                st.markdown(f"[View trace in LangSmith](https://smith.langchain.com/runs/{trace_id})")
                        except Exception as e:
                            with debug_container:
                                st.warning(f"LangSmith finalization error: {str(e)}")
                    
                    # Show completion
                    progress_container.progress(100)
                    progress_text.text("Process complete!")
                    
                    # Display debug info if needed
                    with debug_container:
                        st.write("Final state keys:", list(result.keys()))
                        st.write("Metadata:", result.get("metadata", {}))
                        if "draft" not in result or "review" not in result or "output" not in result:
                            st.error("Some output components are missing!")
                    
                    # Handle agent return status
                    try:
                        # Check if the result contains the expected keys
                        if not result.get("output"):
                            # If output is missing, try to inspect what went wrong
                            st.error("Failed to get output from the multi-agent process")
                            with debug_container:
                                st.write("Available keys in result:", list(result.keys()))
                                st.write("Draft available:", "draft" in result and bool(result["draft"]))
                                st.write("Review available:", "review" in result and bool(result["review"]))
                                st.json(result)
                                
                            # Try to show whatever we got as fallback
                            if result.get("draft"):
                                ai_result = result["draft"]
                                st.warning("Showing draft as fallback since final output wasn't generated")
                            else:
                                ai_result = "No output generated. Please check debug information."
                        else:
                            ai_result = result["output"]
                            st.success("✅ Multi-agent process completed successfully!")
                    
                        # Display results with tabs for all stages
                        tabs = st.tabs(["Final Output", "Initial Draft", "Review Feedback", "Execution Path"])
                        
                        with tabs[0]:
                            st.markdown("### 🛠️ Final AI-Suggested Output")
                            
                            # Format code output
                            if selected_file.endswith('.py'):
                                st.code(ai_result, language='python')
                            else:
                                st.code(ai_result)
                            
                            if st.button("💾 Save AI Result"):
                                try:
                                    with open(file_path, 'w', encoding='utf-8') as f:
                                        f.write(ai_result)
                                    st.success("✅ AI version saved!")
                                except Exception as e:
                                    st.error(f"Error saving AI result: {str(e)}")
                        
                        with tabs[1]:
                            st.markdown("### 📝 Initial Draft")
                            draft = result.get("draft", "No draft was generated")
                            st.code(draft)
                        
                        with tabs[2]:
                            st.markdown("### 🔍 Review Feedback")
                            review = result.get("review", "No review was generated")
                            st.write(review)
                            
                        with tabs[3]:
                            st.markdown("### 🧭 Execution Flow")
                            st.write("The multi-agent system followed this execution path:")
                            
                            # Create a simple visualization of the execution path
                            # (This would be enhanced with actual trace data from LangSmith in a real implementation)
                            st.markdown("""
                            ```mermaid
                            graph TD
                              A[Preprocessor] --> B[Writer Agent]
                              B --> C[Reviewer Agent]
                              C --> D[Optimizer Agent]
                              D --> E{Evaluator}
                              E -->|Complete| F[Final Output]
                              E -->|Refine| C
                            ```
                            """)
                            
                            # Show LangSmith link if available
                            if enable_langsmith_tracing and trace_id:
                                st.markdown(f"[View detailed execution trace in LangSmith](https://smith.langchain.com/runs/{trace_id})")
                            
                    except Exception as e:
                        st.error(f"Error displaying results: {str(e)}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                
                except Exception as e:
                    st.error(f"Error in multi-agent process: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")