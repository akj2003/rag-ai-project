# 1. Use the official Python 3.11 image (Slim version is smaller/faster)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for caching speed)
COPY requirements.txt .

# 4. Install dependencies
# We use --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your application code
COPY . .

# 6. Expose the port Streamlit runs on
EXPOSE 8501

# 7. Check if Streamlit is healthy (Optional but Professional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# 8. The command to run your app
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]