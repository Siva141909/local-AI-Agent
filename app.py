import streamlit as st
import os
from pathlib import Path
from typing import TypedDict, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph
from langsmith import traceable
import traceback

# Set API Keys (better to use st.secrets or environment variables)
os.environ["GOOGLE_API_KEY"] = "AIzaSyDBHqDP8frO-os-FeAlvQc2nb5ENKUhCMw"  # Replace with your actual key
os.environ["LANGCHAIN_API_KEY"] = "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"  # Replace with your actual key
os.environ["LANGCHAIN_PROJECT"] = "LangGraph Multi-Agent"

# Configure Streamlit
st.set_page_config(page_title="Multi-Agent Editor", layout="wide")
st.title("🧠 Multi-Agent File Editor (Gemini + LangGraph)")

# Initialize LLM with error handling
@st.cache_resource
def get_llm():
    try:
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    except Exception as e:
        st.error(f"Failed to initialize LLM: {str(e)}")
        return None

llm = get_llm()

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
        st.subheader("🤖 Multi-Agent Edit using LangGraph")

        prompt = st.text_area("Enter your editing instruction (e.g., 'Refactor the code to improve error handling'):")

        # Multi-agent system
        run_button = st.button("🚀 Run Multi-Agent Edit")
        
        if run_button and prompt and llm:
            with st.spinner("Agents are working on your request..."):
                try:
                    # Define state schema with proper typing
                    class EditState(TypedDict):
                        instruction: str
                        content: str
                        draft: str
                        review: str
                        output: str
                    
                    # Define agents with improved prompts and state handling
                    @traceable
                    def writer_agent(state: EditState) -> Dict[str, Any]:
                        """Writer agent that creates an initial draft based on instructions."""
                        prompt_text = f"""You are a professional developer tasked with editing code based on instructions.
                        
                        ORIGINAL CODE:
                        ```
                        {state['content']}
                        ```
                        
                        INSTRUCTION: {state['instruction']}
                        
                        Please provide an improved version of the code based on the instruction.
                        Only return the complete code without explanations.
                        """
                        
                        response = llm.invoke(prompt_text)
                        # Create new state with all fields to prevent missing keys
                        new_state = state.copy()
                        new_state["draft"] = response.content
                        return new_state
                    
                    @traceable
                    def reviewer_agent(state: EditState) -> Dict[str, Any]:
                        """Reviewer agent that evaluates the draft and provides feedback."""
                        prompt_text = f"""You are a code reviewer examining a draft code change.
                        
                        ORIGINAL CODE:
                        ```
                        {state['content']}
                        ```
                        
                        DRAFT CODE:
                        ```
                        {state['draft']}
                        ```
                        
                        INSTRUCTION: {state['instruction']}
                        
                        Please review the draft carefully and provide specific, actionable feedback:
                        1. Identify any issues or bugs
                        2. Suggest improvements for readability and efficiency
                        3. Point out any missing features relative to the instruction
                        
                        Be concise but comprehensive in your review.
                        """
                        
                        response = llm.invoke(prompt_text)
                        # Create new state with all fields to prevent missing keys
                        new_state = state.copy()
                        new_state["review"] = response.content
                        return new_state
                    
                    @traceable
                    def optimizer_agent(state: EditState) -> Dict[str, Any]:
                        """Optimizer agent that produces the final code based on the review."""
                        prompt_text = f"""You are an expert code optimizer.
                        
                        DRAFT CODE:
                        ```
                        {state['draft']}
                        ```
                        
                        REVIEW FEEDBACK:
                        ```
                        {state['review']}
                        ```
                        
                        INSTRUCTION: {state['instruction']}
                        
                        Based on the review feedback, please provide an optimized final version of the code.
                        Return ONLY the complete, improved code without any explanations or comments about what you changed.
                        """
                        
                        response = llm.invoke(prompt_text)
                        # Create new state with all fields to prevent missing keys
                        new_state = state.copy()
                        new_state["output"] = response.content
                        return new_state

                    # Create progress indicators
                    progress_container = st.empty()
                    progress_text = st.empty()
                    
                    # Debug container
                    debug_container = st.expander("Debug Information", expanded=False)
                    
                    # Display progress for each step
                    def state_callback(state, event_type, node):
                        if event_type == "enter":
                            progress_text.text(f"Running {node}...")
                            if node == "writer":
                                progress_container.progress(33)
                            elif node == "reviewer":
                                progress_container.progress(66)
                            elif node == "optimizer":
                                progress_container.progress(100)
                            
                            # Log state for debugging
                            with debug_container:
                                st.write(f"Current node: {node}")
                                st.write(f"State keys: {state.keys()}")
                    
                    # Create progress display function
                    def update_progress(step):
                        if step == "writer":
                            progress_container.progress(33)
                            progress_text.text("Writer agent creating draft...")
                        elif step == "reviewer":
                            progress_container.progress(66)
                            progress_text.text("Reviewer agent analyzing draft...")
                        elif step == "optimizer":
                            progress_container.progress(100)
                            progress_text.text("Optimizer agent creating final version...")
                    
                    # Build LangGraph with proper state management
                    builder = StateGraph(EditState)
                    builder.add_node("writer", writer_agent)
                    builder.add_node("reviewer", reviewer_agent)
                    builder.add_node("optimizer", optimizer_agent)
                    
                    # Set flow
                    builder.set_entry_point("writer")
                    builder.add_edge("writer", "reviewer")
                    builder.add_edge("reviewer", "optimizer")
                    builder.set_finish_point("optimizer")
                    
                    # Compile graph (without callbacks parameter)
                    app = builder.compile()
                    
                    # Initialize complete state
                    initial_state = {
                        "instruction": prompt,
                        "content": content,
                        "draft": "",
                        "review": "",
                        "output": ""
                    }
                    
                    # Create progress indicators
                    progress_container = st.empty()
                    progress_text = st.empty()
                    
                    # Debug container
                    debug_container = st.expander("Debug Information", expanded=False)
                    
                    # Show initial progress
                    progress_container.progress(0)
                    progress_text.text("Starting multi-agent process...")
                    
                    # Manual step tracking since we can't use callbacks
                    update_progress("writer")
                    
                    # Run graph
                    result = app.invoke(initial_state)
                    
                    # Show completion
                    progress_container.progress(100)
                    progress_text.text("Process complete!")
                    
                    # Display debug info if needed
                    with debug_container:
                        st.write("Final state keys:", list(result.keys()))
                        if "draft" not in result or "review" not in result or "output" not in result:
                            st.error("Some output components are missing!")
                            st.write("Complete result:", result)
                    
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
                        tabs = st.tabs(["Final Output", "Initial Draft", "Review Feedback"])
                        
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
                            
                    except Exception as e:
                        st.error(f"Error displaying results: {str(e)}")
                        st.error(f"Traceback: {traceback.format_exc()}")
                
                except Exception as e:
                    st.error(f"Error in multi-agent process: {str(e)}")
                    st.error(f"Traceback: {traceback.format_exc()}")