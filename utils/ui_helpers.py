import time
import streamlit as st
from contextlib import contextmanager

@contextmanager
def timer(label="Processing"):
    """
    Context manager to measure and display execution time.
    """
    start_time = time.time()
    yield
    end_time = time.time()
    duration = end_time - start_time
    
    # Display the metric nicely in a green box
    st.success(f"✅ {label} finished in **{duration:.2f} seconds**.")