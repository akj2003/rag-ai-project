import streamlit as st
import os
import base64
import re
from PIL import Image
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from streamlit_mermaid import st_mermaid

# --- PAGE CONFIG ---
st.set_page_config(page_title="Social Media Engine", page_icon="📱", layout="wide")
st.title("📱 AI Social Media & Branding Engine")

st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
        <b>Personal Branding Automation:</b><br>
        Generate high-engagement thought leadership for LinkedIn. Optionally generate professional flow 
        diagrams to accompany your posts. Or, use the Instagram tab to analyze images for captions.
    </div>
    <br>
""", unsafe_allow_html=True)

# --- AI CONFIGURATION ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

@st.cache_resource
def get_text_llm():
    return ChatOllama(model="llama3.2", temperature=0.7, base_url=ollama_url)

@st.cache_resource
def get_vision_llm():
    return ChatOllama(model="llava", temperature=0.7, base_url=ollama_url)

# --- STATE MANAGEMENT ---
if "linkedin_post" not in st.session_state:
    st.session_state.linkedin_post = ""
if "mermaid_code" not in st.session_state:
    st.session_state.mermaid_code = ""

# --- UI TABS ---
tab1, tab2 = st.tabs(["💼 LinkedIn (Thought Leadership)", "📸 Instagram (Visual Storytelling)"])

# ==========================================
# TAB 1: LINKEDIN GENERATOR
# ==========================================
with tab1:
    st.subheader("1. Draft the Content")
    tech_trend = st.text_input("Enter a Technology Trend or Topic (e.g., 'Agentic AI Workflows', 'RAG Pipelines'):")
    tone = st.selectbox("Select Tone:", ["Strategic & Executive", "Educational & Technical", "Inspirational / Career Growth"])
    
    if st.button("📝 Generate LinkedIn Post"):
        if not tech_trend:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Drafting your thought leadership post..."):
                try:
                    llm = get_text_llm()
                    prompt = PromptTemplate.from_template("""
                    You are a Strategic IT Leader and Associate Director with 15+ years of experience in enterprise modernization, 
                    agile transformations, and AI integration. 
                    
                    Write a highly engaging LinkedIn post about: {topic}
                    
                    Tone: {tone}
                    
                    Structure:
                    1. A strong, scroll-stopping hook (1-2 sentences).
                    2. The core insight or business value (use bullet points if appropriate).
                    3. A clear takeaway or question to drive comments and engagement.
                    4. Exactly 5-7 highly relevant hashtags at the bottom.
                    
                    Do NOT include generic greetings. Make it sound authentic, experienced, and authoritative.
                    """)
                    
                    chain = prompt | llm
                    response = chain.invoke({"topic": tech_trend, "tone": tone})
                    
                    st.session_state.linkedin_post = response.content
                    st.session_state.mermaid_code = "" # Reset any old diagrams
                    st.success("Draft Generated!")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    # Display the generated post if it exists in session state
    if st.session_state.linkedin_post:
        st.text_area("Your Post (Edit before copying):", value=st.session_state.linkedin_post, height=300)
        
        st.markdown("---")
        st.subheader("2. Visual Assets (Optional)")
        st.write("Visuals increase LinkedIn engagement by over 200%. Let the AI map this concept into a flow diagram.")
        
       # The Optional Diagram Button
        if st.button("📊 Generate Flow Diagram"):
            with st.spinner("Translating post concepts into a visual flowchart..."):
                try:
                    llm = get_text_llm()
                    diagram_prompt = PromptTemplate.from_template("""
                    Analyze the following LinkedIn post and create a Mermaid.js Flowchart (graph TD) that visually represents the core concepts, architecture, or workflow described.
                    
                    LinkedIn Post:
                    {post_content}
                    
                    Rules:
                    1. Output ONLY the Mermaid.js code block. No markdown, no conversational text.
                    2. Start immediately with 'graph TD'.
                    3. DO NOT use parentheses () or special characters inside node descriptions unless you wrap the description in double quotes like this: A["Agentic AI (LLM)"]
                    4. Keep it simple: maximum 8 nodes.
                    """)
                    
                    diagram_chain = diagram_prompt | llm
                    raw_diagram = diagram_chain.invoke({"post_content": st.session_state.linkedin_post}).content
                    
                    # --- CLEAN THE CODE ---
                    clean_code = raw_diagram.replace("```mermaid", "").replace("```", "").strip()
                    st.session_state.mermaid_code = clean_code
                    
                except Exception as e:
                    st.error(f"Diagram generation failed: {e}")

    # Display the diagram if it exists
    if st.session_state.mermaid_code:
        st.success("Diagram generated! Right-click the image below to save it for LinkedIn.")
        
        with st.expander("🛠️ View/Edit Raw Mermaid Code (If diagram has a syntax error)"):
            st.code(st.session_state.mermaid_code, language="mermaid")
        
        st.markdown("---")
        
        # --- THE BULLETPROOF IMAGE RENDERER ---
        try:
            # Encode the raw mermaid code into Base64 format
            graphbytes = st.session_state.mermaid_code.encode("utf-8")
            base64_bytes = base64.b64encode(graphbytes)
            base64_string = base64_bytes.decode("utf-8")
            
            # Pass it to the Mermaid.ink API to get a pure PNG image back
            image_url = f"https://mermaid.ink/img/{base64_string}"
            
            # Display it as a standard Streamlit image
            st.image(image_url, caption="AI-Generated Architecture Flow", use_column_width=True)
            
        except Exception as e:
            st.error("Failed to render the image. Check the raw code in the expander for syntax errors.")
            
# ==========================================
# TAB 2: INSTAGRAM CAPTION GENERATOR
# ==========================================
with tab2:
    st.subheader("Generate an Instagram Caption from an Image")
    uploaded_image = st.file_uploader("Upload a photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    insta_vibe = st.selectbox("Caption Vibe:", ["Professional / Behind the Scenes", "Casual & Fun", "Motivational / Fitness"])
    
    if st.button("Generate Caption"):
        if not uploaded_image:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Analyzing image and generating caption..."):
                try:
                    image_bytes = uploaded_image.read()
                    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
                    
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    llm = get_vision_llm()
                    
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": f"You are an expert social media manager. Look at this image and write an engaging Instagram caption for it. The vibe should be: {insta_vibe}. Include emojis and 5-8 relevant hashtags to maximize reach."},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    )
                    
                    response = llm.invoke([message])
                    
                    st.success("Caption Generated!")
                    st.text_area("Copy your caption:", value=response.content, height=200)
                except Exception as e:
                    st.error(f"Vision analysis failed: {e}. (Ensure you have pulled the 'llava' model in your backend).")