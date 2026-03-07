import streamlit as st
from fpdf import FPDF
import tempfile
import os
from datetime import datetime
from streamlit_quill import st_quill

# LangChain & RAG Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate

# --- PAGE CONFIG ---
st.set_page_config(page_title="Cover Letter Generator", page_icon="📄")
st.title("📄 AI-Powered Cover Letter Generator")

st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
        <b>AI Document Engine (RAG, ATS & Rich Text):</b><br>
        1. Upload your Resume and paste a Job Description.<br>
        2. Let RAG generate a highly targeted draft.<br>
        3. Use the <b>Rich Text Editor</b> to add bullet points and bold formatting before exporting to PDF.
    </div>
    <br>
""", unsafe_allow_html=True)

# --- AI CONFIGURATION ---
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0.6, base_url=ollama_url)

@st.cache_resource
def get_embeddings():
    os.environ["OLLAMA_HOST"] = ollama_url
    return OllamaEmbeddings(model="nomic-embed-text")

# --- STATE MANAGEMENT ---
if "ai_generated_body" not in st.session_state:
    st.session_state.ai_generated_body = ""

# --- USER INPUTS ---
st.subheader("1. Personal Details")
user_name = st.text_input("Full Name:", value="Aswin Kumar Jegadeesh")

col_contact1, col_contact2 = st.columns(2)
with col_contact1:
    user_contact = st.text_input("Contact Info:", value="New Jersey, USA | +1 973-294-2525 | achuj2003@gmail.com")
with col_contact2:
    user_linkedin = st.text_input("LinkedIn Profile:", value="linkedin.com/in/akj2003/")

st.subheader("2. Target Job & Resume Ingestion")
col_job1, col_job2 = st.columns(2)
with col_job1:
    target_role = st.text_input("Target Role:", value=" ")
with col_job2:
    target_company = st.text_input("Target Company:", value=" ")

uploaded_resume = st.file_uploader("Upload your current Resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste Job Description Here:", height=150)

# --- RAG & AI GENERATION BUTTON ---
if st.button("✨ Auto-Write with RAG & Llama 3.2"):
    if not job_description or not uploaded_resume:
        st.warning("⚠️ Please upload a Resume and paste a Job Description to proceed.")
    else:
        with st.spinner("Ingesting Resume into ChromaDB & Generating Letter..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_resume.read())
                    temp_pdf_path = tmp_file.name

                loader = PyPDFLoader(temp_pdf_path)
                docs = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                splits = text_splitter.split_documents(docs)

                embeddings = get_embeddings()
                vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                relevant_chunks = retriever.invoke(job_description)
                resume_context = "\n\n".join([doc.page_content for doc in relevant_chunks])

                llm = get_llm()
                prompt = PromptTemplate.from_template("""
                You are an expert executive career coach. Write the body paragraphs for a cover letter.
                
                Target Role: {role}
                Target Company: {company}
                
                Candidate's Relevant Experience (Retrieved via RAG from their Resume):
                {context}
                
                Job Description to align with:
                {jd}
                
                Instructions:
                - Write ONLY the body paragraphs. 
                - DO NOT include "Dear Hiring Manager", addresses, or "Sincerely/Warm regards". 
                - Use HTML tags like <b> for bolding key metrics and <ul><li> for lists if appropriate.
                - Keep it to 3 or 4 powerful, concise paragraphs.
                - Tone: Confident, executive, and consultative.
                """)
                
                chain = prompt | llm
                response = chain.invoke({
                    "role": target_role,
                    "company": target_company,
                    "context": resume_context,
                    "jd": job_description
                })
                
                # Format the AI output as HTML so Quill reads it properly
                raw_text = response.content
                if not raw_text.startswith("<"):
                    raw_text = raw_text.replace("\n\n", "</p><p>")
                    raw_text = f"<p>{raw_text}</p>"
                
                st.session_state.ai_generated_body = raw_text
                st.success("Draft generated from your Resume! Review and edit below.")
                
                vectorstore.delete_collection()
                os.remove(temp_pdf_path)
                
            except Exception as e:
                st.error(f"RAG Pipeline failed: {e}")

st.subheader("3. Core Message (Rich Text Editor)")
st.info("💡 Use the toolbar to add bold text, italics, or bullet points. The PDF will match your formatting.")

# --- THE UX UPGRADE: QUILL RICH TEXT EDITOR ---
dynamic_body_html = st_quill(
    value=st.session_state.ai_generated_body,
    html=True,  # Tells Quill to output HTML
    placeholder="Write or edit your cover letter body here...",
    key="quill_editor"
)

# --- ATS ANALYZER ---
st.markdown("---")
st.subheader("🔍 ATS Keyword Match Analyzer")

if st.button("📊 Calculate ATS Score"):
    if not job_description:
        st.warning("Please paste a Job Description at the top first.")
    elif not dynamic_body_html:
        st.warning("The cover letter body is empty. Generate or write a draft first.")
    else:
        with st.spinner("Analyzing keyword alignment..."):
            try:
                llm = get_llm()
                ats_prompt = PromptTemplate.from_template("""
                You are a strict Corporate Applicant Tracking System (ATS) algorithm.
                Compare the Job Description to the Candidate's Cover Letter HTML text.
                
                Job Description:
                {jd}
                
                Cover Letter (HTML):
                {cover_letter}
                
                Evaluate the match and respond STRICTLY in this format:
                
                ### **ATS Match Score:** [Insert Percentage e.g., 85%]
                
                **✅ Key Alignments:**
                - [Bullet point 1]
                - [Bullet point 2]
                
                **⚠️ Missing Keywords to Add:**
                - [Keyword/Phrase 1 from JD not found in cover letter]
                - [Keyword/Phrase 2 from JD not found in cover letter]
                """)
                
                ats_chain = ats_prompt | llm
                ats_response = ats_chain.invoke({
                    "jd": job_description,
                    "cover_letter": dynamic_body_html
                })
                
                st.info(ats_response.content)
            except Exception as e:
                st.error(f"ATS Analysis failed: {e}")

# --- PDF GENERATION LOGIC (UPGRADED FOR HTML) ---
class FormalLetterPDF(FPDF):
    def __init__(self, name, contact, linkedin, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user_name = name
        self.user_contact = contact
        self.user_linkedin = linkedin

    def header(self):
        self.set_font('Arial', 'B', 18)
        self.set_text_color(0, 51, 102)
        self.cell(0, 10, self.user_name, 0, 1, 'C')
        
        self.set_font('Arial', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, self.user_contact, 0, 1, 'C')
        self.cell(0, 5, self.user_linkedin, 0, 1, 'C')
        
        self.set_draw_color(200, 200, 200)
        self.line(10, 32, 200, 32)
        self.ln(15)

def assemble_pdf(name, contact, linkedin, role, company, body_html):
    pdf = FormalLetterPDF(name, contact, linkedin)
    pdf.add_page()
    
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)
    
    # 1. Add Current Date
    current_date = datetime.today().strftime('%B %d, %Y')
    pdf.cell(0, 10, current_date, 0, 1)
    pdf.ln(5)
    
    # 2. Add Salutation
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 10, "Dear Hiring Manager,", 0, 1)
    pdf.ln(5)
    
    # 3. Add Dynamic Body Text (Using fpdf2's HTML engine)
    # This reads the <b>, <i>, and <ul> tags from Quill and renders them perfectly!
    pdf.write_html(body_html)
            
    # 4. Add Sign-off
    pdf.ln(5)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 6, "Warm regards,", 0, 1)
    pdf.ln(8)
    pdf.set_font('Arial', 'B', 11)
    pdf.cell(0, 6, name, 0, 1)
    
    temp_dir = tempfile.mkdtemp()
    safe_name = name.replace(' ', '_')
    safe_company = company.replace(' ', '')
    file_name = f"{safe_name}_CoverLetter_{safe_company}.pdf"
    
    temp_filepath = os.path.join(temp_dir, file_name)
    pdf.output(temp_filepath)
    
    return temp_filepath, file_name

# --- EXPORT / DOWNLOAD UI ---
st.markdown("---")
st.subheader("4. Generate & Download")

if st.button("Generate Structured PDF"):
    with st.spinner("Assembling document..."):
        try:
            filepath, output_filename = assemble_pdf(
                user_name, user_contact, user_linkedin, target_role, target_company, dynamic_body_html
            )
            
            with open(filepath, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
                
            st.success("✨ PDF assembled and formatted successfully!")
            
            st.download_button(
                label="⬇️ Download Formal Cover Letter",
                data=pdf_bytes,
                file_name=output_filename,
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}. Check your HTML formatting.")