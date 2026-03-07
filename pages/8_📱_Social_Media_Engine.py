import streamlit as st
import os
import base64
from PIL import Image
from io import BytesIO
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# --- PAGE CONFIG ---
st.set_page_config(page_title="Social Media Engine", page_icon="📱")
st.title("📱 AI Social Media & Branding Engine")

st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
        <b>Personal Branding Automation:</b><br>
        Generate high-engagement thought leadership for LinkedIn, or upload an image to generate 
        aesthetic, context-aware Instagram captions with targeted hashtags.
    </div>
    <br>
""", unsafe_allow_html=True)

# --- AI CONFIGURATION ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

@st.cache_resource
def get_text_llm():
    # Llama 3.2 is perfect for writing the LinkedIn text
    return ChatOllama(model="llama3.2", temperature=0.7, base_url=ollama_url)

@st.cache_resource
def get_vision_llm():
    # We use 'llava' (Large Language-and-Vision Assistant) to look at the Instagram photos
    return ChatOllama(model="llava", temperature=0.7, base_url=ollama_url)

# --- UI TABS ---
tab1, tab2 = st.tabs(["💼 LinkedIn (Thought Leadership)", "📸 Instagram (Visual Storytelling)"])

# ==========================================
# TAB 1: LINKEDIN GENERATOR
# ==========================================
with tab1:
    st.subheader("Generate a LinkedIn Post")
    tech_trend = st.text_input("Enter a Technology Trend or Topic (e.g., 'Agentic AI in Enterprise', 'Cloud Cost Optimization'):")
    
    tone = st.selectbox("Select Tone:", ["Strategic & Executive", "Educational & Technical", "Inspirational / Career Growth"])
    
    if st.button("Generate LinkedIn Post"):
        if not tech_trend:
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Drafting your thought leadership post..."):
                try:
                    llm = get_text_llm()
                    
                    # We inject an executive persona into the prompt so it sounds like a senior leader
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
                    
                    st.success("Draft Generated!")
                    st.text_area("Edit your post before copying:", value=response.content, height=350)
                except Exception as e:
                    st.error(f"Generation failed: {e}")

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
                    # 1. Read and display the image
                    image_bytes = uploaded_image.read()
                    st.image(image_bytes, caption="Uploaded Image", use_column_width=True)
                    
                    # 2. Encode image to Base64 so the AI can "see" it
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    
                    # 3. Prepare the multimodal message for the Vision model
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