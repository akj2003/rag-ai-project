import streamlit as st
import os
import base64
import requests
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
        
        # --- THE DRAW.IO GENERATOR ---
        if st.button("📊 Generate Draw.io Diagram"):
            with st.spinner("Extracting concepts and building Draw.io architecture..."):
                try:
                    llm = get_text_llm()
                    
                    # 1. Ask the AI for a simple list of concepts
                    diagram_prompt = PromptTemplate.from_template("""
                    Read this LinkedIn post and extract a linear sequence of 3 to 5 core concepts or steps.
                    Output ONLY a single line of text, with each step separated by a pipe character (|).
                    
                    Example Output:
                    Legacy Tech Debt | AI Integration Strategy | Automated Pipelines | Massive ROI
                    
                    Do not add numbers, bullets, markdown, or explanations. Just the pipe-separated text.
                    
                    Post: {post_content}
                    """)
                    
                    diagram_chain = diagram_prompt | llm
                    raw_steps = diagram_chain.invoke({"post_content": st.session_state.linkedin_post}).content
                    
                    # 2. Clean the AI's output
                    cleaned_string = raw_steps.replace('`', '').replace('**', '').replace('\n', '')
                    steps = [step.strip() for step in cleaned_string.split('|') if step.strip()]
                    
                    # 3. Mathematically build the Draw.io XML structure
                    if len(steps) > 1:
                        # Draw.io required XML headers
                        xml_content = [
                            '<mxfile host="app.diagrams.net" agent="Streamlit">',
                            '  <diagram id="ai_flow" name="LinkedIn_Architecture">',
                            '    <mxGraphModel dx="1000" dy="1000" grid="1" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" math="0" shadow="0">',
                            '      <root>',
                            '        <mxCell id="0" />',
                            '        <mxCell id="1" parent="0" />'
                        ]
                        
                        x_position = 40
                        # Build Nodes and Edges (Horizontal layout, great for LinkedIn graphics)
                        for i, step in enumerate(steps):
                            node_id = f"node_{i}"
                            # XML escape characters to prevent corruption
                            safe_text = step.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                            
                            # Add the Node (styled with a professional corporate blue)
                            xml_content.append(f'        <mxCell id="{node_id}" value="{safe_text}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;fontSize=14;" vertex="1" parent="1">')
                            xml_content.append(f'          <mxGeometry x="{x_position}" y="100" width="200" height="60" as="geometry" />')
                            xml_content.append('        </mxCell>')
                            
                            # Add the Arrow connecting to the previous node
                            if i > 0:
                                prev_id = f"node_{i-1}"
                                edge_id = f"edge_{i}"
                                xml_content.append(f'        <mxCell id="{edge_id}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;" edge="1" parent="1" source="{prev_id}" target="{node_id}">')
                                xml_content.append('          <mxGeometry relative="1" as="geometry" />')
                                xml_content.append('        </mxCell>')
                                
                            x_position += 280 # Space out the nodes horizontally
                            
                        # Close the XML tags
                        xml_content.extend([
                            '      </root>',
                            '    </mxGraphModel>',
                            '  </diagram>',
                            '</mxfile>'
                        ])
                        
                        st.session_state.drawio_code = "\n".join(xml_content)
                        st.success("✨ Draw.io file successfully compiled!")
                    else:
                        st.error("The AI failed to extract distinct steps. Try generating the text post again.")
                        st.session_state.drawio_code = ""
                        
                except Exception as e:
                    st.error(f"Draw.io compilation failed: {e}")

    # --- RENDER THE DOWNLOAD BUTTON ---
    if "drawio_code" in st.session_state and st.session_state.drawio_code:
        st.markdown("### 📥 Export Architecture Diagram")
        st.info("Your diagram is ready! Download the file below, drag it into **[app.diagrams.net](https://app.diagrams.net/)**, and export it as a high-res PNG for your post.")
        
        # Convert string to bytes for the download button
        drawio_bytes = st.session_state.drawio_code.encode('utf-8')
        
        st.download_button(
            label="⬇️ Download .drawio File",
            data=drawio_bytes,
            file_name="LinkedIn_Architecture_Flow.drawio",
            mime="application/xml"
        )
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