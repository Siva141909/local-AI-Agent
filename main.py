import streamlit as st
import os
from pathlib import Path

# Import directly without using dotenv
# This code doesn't rely on the dotenv package
import os

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyDBHqDP8frO-os-FeAlvQc2nb5ENKUhCMw"

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# No need for load_dotenv() here
# API key is already set above

# Gemini model setup with correct model name
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

st.set_page_config(page_title="Local AI Agent", layout="wide")
st.title("🧠 Local AI Agent Interface (File Explorer + Gemini Edit)")

# Directory input
folder_path = st.sidebar.text_input("📁 Enter directory path:", value=str(Path.cwd()))

# Filter for editable files
def list_files(path):
    return [f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith(('.py', '.txt', '.md', '.json', '.csv'))]

if not os.path.isdir(folder_path):
    st.error("🚫 Invalid directory path.")
else:
    files = list_files(folder_path)

    if not files:
        st.info("📂 No editable files found.")
    else:
        selected_file = st.sidebar.selectbox("Select file to edit", files)
        file_path = os.path.join(folder_path, selected_file)

        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        st.markdown(f"### ✏️ Editing: `{selected_file}`")
        new_content = st.text_area("Edit the content below:", value=content, height=400)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Changes"):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                st.success("✅ Saved!")

        with col2:
            with open(file_path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download",
                    data=f,
                    file_name=selected_file,
                    mime="text/plain"
                )

        # AI Edit Prompt
        st.divider()
        st.subheader("🤖 Gemini Edit")
        prompt = st.text_area("Enter your prompt (e.g., 'Optimize this code'):")

        if st.button("✨ Run AI Edit") and prompt:
            full_prompt = f"You are an expert developer. {prompt}\n\nFile Content:\n{content}"
            
            try:
                response = llm.invoke(full_prompt)
                
                if hasattr(response, 'content'):
                    ai_result = response.content
                else:
                    ai_result = str(response)
                    
                st.markdown("### 🔧 AI-Suggested Edit")
                st.code(ai_result, language='python')
                
                if st.button("💾 Save AI Result"):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(ai_result)
                    st.success("✅ AI version saved!")
            except Exception as e:
                st.error(f"Error calling Gemini: {str(e)}")
                st.info("Please check your API key and try again.")