import streamlit as st
import os
import base64
import requests
import re
import boto3
import json
from PIL import Image
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

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
        
        # --- NEW: AWS BEDROCK RICH IMAGE GENERATOR ---
        st.markdown("---")
        st.markdown("### 🎨 Generate Rich Visuals (AWS Bedrock)")
        st.write("Generate a high-fidelity conceptual image using Amazon Titan or Stable Diffusion.")
        
        if st.button("🌌 Generate Rich Image"):
            with st.spinner("Commanding Llama 3.2 to write the prompt, and Bedrock to render the image..."):
                try:
                    # 1. Use local Llama 3.2 to write a highly optimized image generation prompt
                    llm = get_text_llm()
                    img_prompt_template = PromptTemplate.from_template("""
                    You are an expert AI image prompt engineer. Read this LinkedIn post and write a highly detailed, 
                    professional, and corporate image generation prompt that perfectly captures the core theme.
                    
                    Rules:
                    1. Keep it under 400 characters.
                    2. Specify a professional, high-tech corporate aesthetic (e.g., "cinematic lighting, sleek 3D render, enterprise technology").
                    3. DO NOT ask for any text, letters, or words in the image (AI is bad at spelling).
                    4. Output ONLY the raw prompt.
                    
                    Post: {post_content}
                    """)
                    
                    img_chain = img_prompt_template | llm
                    generated_prompt = img_chain.invoke({"post_content": st.session_state.linkedin_post}).content.strip()
                    
                    st.info(f"**Optimized Prompt sent to AWS:** {generated_prompt}")
                    
                    # 2. Call AWS Bedrock to generate the image
                    # Note: Ensure your region matches where you requested Bedrock model access (e.g., us-east-1)
                    bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1') 
                    
                    # Payload configured for Amazon Titan Image Generator v1
                    payload = {
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {
                            "text": generated_prompt
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": 1,
                            "quality": "standard",
                            "cfgScale": 8.0,
                            "height": 1024,
                            "width": 1024,
                            "seed": 0 # Random seed for unique images
                        }
                    }
                    
                    response = bedrock_client.invoke_model(
                        modelId="amazon.titan-image-generator-v1",
                        contentType="application/json",
                        accept="application/json",
                        body=json.dumps(payload)
                    )
                    
                    # 3. Decode and Display the Image
                    response_body = json.loads(response.get('body').read())
                    base64_image = response_body.get('images')[0]
                    image_bytes = base64.b64decode(base64_image)
                    
                    st.success("✨ Rich image successfully generated by AWS Bedrock!")
                    st.image(image_bytes, caption="Generated via Amazon Titan", use_container_width=True)
                    
                    st.download_button(
                        label="⬇️ Download High-Res Image",
                        data=image_bytes,
                        file_name="LinkedIn_Rich_Graphic.png",
                        mime="image/png"
                    )
                    
                except Exception as e:
                    st.error(f"AWS Bedrock generation failed. Did you enable Model Access in the AWS Console? Error details: {e}")
                    
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
                    st.image(image_bytes, caption="Generated via Amazon Titan", use_container_width=True)
                    
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