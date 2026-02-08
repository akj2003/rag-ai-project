import streamlit as st
from PIL import Image
import io
import base64
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# --- IMPORT TIMER ---
from utils.ui_helpers import timer

# --- PAGE CONFIG ---
st.set_page_config(page_title="Vision Agent", page_icon="👁️")
st.title("👁️ Vision Agent (Turbo Mode)")

# --- GET OLLAMA URL (Docker Support) ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- OPTIMIZATION FUNCTIONS ---
def resize_image(image, target_width):
    """
    Resizes image to target_width.
    512px = ~4x faster than 1024px.
    """
    width, height = image.size
    ratio = target_width / width
    new_height = int(height * ratio)
    return image.resize((target_width, new_height), Image.Resampling.LANCZOS)

def get_base64_image(image):
    """
    Aggressively compresses image to JPEG (70% quality).
    """
    img_byte_arr = io.BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    
    # Lower quality = Faster processing
    image.save(img_byte_arr, format='JPEG', quality=70)
    return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚡ Performance Settings")
    
    # Default to "Turbo" to prevent crashing
    mode = st.radio(
        "Speed vs Quality:",
        ["Turbo (512px) - Fastest", "Balanced (800px)", "High (1024px)"]
    )
    
    if mode == "Turbo (512px) - Fastest":
        target_width = 512
    elif mode == "Balanced (800px)":
        target_width = 800
    else:
        target_width = 1024

    st.markdown("---")
    if st.button("🧹 Force Release RAM"):
        # Dummy call to unload model
        try:
            ChatOllama(base_url=ollama_url, model="llama3.2-vision", keep_alive=0).invoke("bye")
            st.success("RAM Freed!")
        except:
            st.warning("Could not connect to Ollama.")

# --- MAIN LOGIC ---
st.info(f"Connected to AI at: `{ollama_url}`")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # 1. Load Image
    original_image = Image.open(uploaded_file)
    
    # 2. Optimize
    optimized_image = resize_image(original_image, target_width)
    
    # Show comparison stats
    orig_size = f"{original_image.size[0]}x{original_image.size[1]}"
    new_size = f"{optimized_image.size[0]}x{optimized_image.size[1]}"
    st.caption(f"📉 Resized: {orig_size} ➝ **{new_size}** (Faster Processing)")
    
    st.image(optimized_image, caption="What the AI sees", use_column_width=True)
    
    # 3. Convert to Base64
    img_base64 = get_base64_image(optimized_image)

    # 4. Chat
    st.write("### Ask the Vision Agent")
    user_question = st.text_input("Question:", placeholder="Describe this image...")

    if user_question:
        with st.spinner("👁️ Reading Image..."):
            try:
                # --- TIMER 1: MODEL LOADING ---
                with timer("Model Loading"):
                    llm = ChatOllama(
                        model="moondream",  # <--- CHANGED FROM llama3.2-vision
                        temperature=0,
                        base_url=ollama_url,
                        num_ctx=2048,
                        keep_alive="0m" 
                    )
                
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": user_question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                    ]
                )
                
                # --- TIMER 2: GENERATION ---
                with timer("AI Thinking"):
                    response = llm.invoke([message])
                
                st.write(response.content)
                
            except Exception as e:
                st.error(f"Error: {e}")