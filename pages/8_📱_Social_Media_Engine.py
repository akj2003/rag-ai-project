import streamlit as st
import os
import base64
import json
import boto3
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Social Media Engine", page_icon="📱", layout="wide")
st.title("📱 AI Social Media & Branding Engine")

st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
        <b>Personal Branding Automation:</b><br>
        Generate high-engagement thought leadership for LinkedIn. Optionally export professional 
        Draw.io architectures, generate rich images with AWS Bedrock, or caption Instagram photos.
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

# --- UI TABS ---
tab1, tab2 = st.tabs(["💼 LinkedIn (Thought Leadership)", "📸 Instagram (Visual Storytelling)"])

# ==========================================
# TAB 1: LINKEDIN GENERATOR
# ==========================================
with tab1:
    st.subheader("1. Draft the Content")
    tech_trend = st.text_input("Enter a Technology Trend or Topic (e.g., 'Agentic AI Workflows'):")
    tone = st.selectbox("Select Tone:", ["Strategic & Executive", "Educational & Technical", "Inspirational / Career Growth"])
    
    if st.button("📝 Generate LinkedIn Post"):
        if not tech_trend:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Drafting your thought leadership post..."):
                try:
                    llm = get_text_llm()
                    prompt = PromptTemplate.from_template("""
                    You are a Strategic IT Leader and Associate Director with 15+ years of experience. 
                    Write a highly engaging LinkedIn post about: {topic}
                    Tone: {tone}
                    Structure:
                    1. A strong, scroll-stopping hook.
                    2. The core insight or business value.
                    3. A clear takeaway or question.
                    4. Exactly 5-7 highly relevant hashtags.
                    """)
                    
                    chain = prompt | llm
                    response = chain.invoke({"topic": tech_trend, "tone": tone})
                    st.session_state.linkedin_post = response.content
                    # Clear out old diagrams/images on new post generation
                    st.session_state.pop("drawio_code", None)
                    st.success("Draft Generated!")
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    # Display the generated post
    if st.session_state.linkedin_post:
        st.text_area("Your Post (Edit before copying):", value=st.session_state.linkedin_post, height=300)
        
        st.markdown("---")
        st.subheader("2. Visual Assets (Choose One)")
        
        col_drawio, col_bedrock = st.columns(2)
        
        # --- OPTION A: DRAW.IO ARCHITECTURE ---
        with col_drawio:
            st.markdown("#### 📐 Technical Architecture")
            st.write("Generate a Draw.io structural flow diagram.")
            if st.button("📊 Generate Draw.io Diagram"):
                with st.spinner("Building Draw.io architecture..."):
                    try:
                        llm = get_text_llm()
                        diagram_prompt = PromptTemplate.from_template("""
                        Read this post and extract a linear sequence of 3 to 5 core concepts.
                        Output ONLY a single line of text, separated by a pipe (|).
                        Example: Tech Debt | AI Strategy | Automation | ROI
                        Post: {post_content}
                        """)
                        raw_steps = (diagram_prompt | llm).invoke({"post_content": st.session_state.linkedin_post}).content
                        
                        cleaned_string = raw_steps.replace('`', '').replace('**', '').replace('\n', '')
                        steps = [step.strip() for step in cleaned_string.split('|') if step.strip()]
                        
                        if len(steps) > 1:
                            xml_content = [
                                '<mxfile host="app.diagrams.net" agent="Streamlit">',
                                '  <diagram id="ai_flow" name="LinkedIn_Architecture">',
                                '    <mxGraphModel dx="1000" dy="1000" grid="1" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" math="0" shadow="0">',
                                '      <root>',
                                '        <mxCell id="0" />',
                                '        <mxCell id="1" parent="0" />'
                            ]
                            x_pos = 40
                            for i, step in enumerate(steps):
                                node_id = f"node_{i}"
                                safe_text = step.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                                xml_content.append(f'        <mxCell id="{node_id}" value="{safe_text}" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#6c8ebf;fontStyle=1;fontSize=14;" vertex="1" parent="1"><mxGeometry x="{x_pos}" y="100" width="200" height="60" as="geometry" /></mxCell>')
                                if i > 0:
                                    prev_id = f"node_{i-1}"
                                    xml_content.append(f'        <mxCell id="edge_{i}" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;" edge="1" parent="1" source="{prev_id}" target="{node_id}"><mxGeometry relative="1" as="geometry" /></mxCell>')
                                x_pos += 280
                                
                            xml_content.extend(['      </root></mxGraphModel></diagram></mxfile>'])
                            st.session_state.drawio_code = "\n".join(xml_content)
                            st.success("✨ Draw.io compiled!")
                        else:
                            st.error("Failed to extract steps.")
                    except Exception as e:
                        st.error(f"Draw.io failed: {e}")

            if "drawio_code" in st.session_state and st.session_state.drawio_code:
                st.download_button("⬇️ Download .drawio File", data=st.session_state.drawio_code.encode('utf-8'), file_name="LinkedIn_Architecture.drawio", mime="application/xml")
                st.caption("Drag this file into [app.diagrams.net](https://app.diagrams.net/)")

        # --- OPTION B: AWS BEDROCK RICH IMAGE ---
        with col_bedrock:
            st.markdown("#### 🎨 Rich Visual Graphic")
            st.write("Generate a high-fidelity image using Amazon Bedrock.")
            if st.button("🌌 Generate Rich Image"):
                with st.spinner("Prompting AWS Bedrock..."):
                    try:
                        llm = get_text_llm()
                        img_prompt_template = PromptTemplate.from_template("""
                        Write a highly detailed, professional image generation prompt based on this post.
                        Rules: Under 400 characters. Corporate aesthetic. NO text/words in the image. ONLY output the prompt.
                        Post: {post_content}
                        """)
                        generated_prompt = (img_prompt_template | llm).invoke({"post_content": st.session_state.linkedin_post}).content.strip()
                        
                        st.info(f"**Prompt:** {generated_prompt}")
                        
                        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1') 
                        payload = {
                            "taskType": "TEXT_IMAGE",
                            "textToImageParams": {"text": generated_prompt},
                            "imageGenerationConfig": {"numberOfImages": 1, "quality": "standard", "cfgScale": 8.0, "height": 1024, "width": 1024, "seed": 0}
                        }
                        
                        response = bedrock_client.invoke_model(
                            modelId="amazon.titan-image-generator-v2:0",
                            contentType="application/json",
                            accept="application/json",
                            body=json.dumps(payload)
                        )
                        
                        response_body = json.loads(response.get('body').read())
                        image_bytes = base64.b64decode(response_body.get('images')[0])
                        
                        st.success("✨ Image generated!")
                        # FIXED: Changed from use_column_width to use_container_width to clear the yellow warning
                        st.image(image_bytes, caption="Generated via Amazon Titan", use_container_width=True)
                        st.download_button("⬇️ Download Image", data=image_bytes, file_name="LinkedIn_Graphic.png", mime="image/png")
                    except Exception as e:
                        st.error(f"AWS Bedrock failed. Check IAM permissions. Error: {e}")

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
                    # FIXED: Changed to use_container_width
                    st.image(image_bytes, caption="Uploaded Image", use_container_width=True)
                    
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    llm = get_vision_llm()
                    
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": f"Write an engaging Instagram caption for this image. Vibe: {insta_vibe}. Include emojis and hashtags."},
                            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                        ]
                    )
                    
                    response = llm.invoke([message])
                    st.success("Caption Generated!")
                    st.text_area("Copy your caption:", value=response.content, height=200)
                except Exception as e:
                    st.error(f"Vision analysis failed: {e}. Ensure you pulled 'llava'.")