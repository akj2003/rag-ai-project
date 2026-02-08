import streamlit as st
import psutil
import os
import time
import pandas as pd

def get_process_df():
    """
    Fetches the top processes by memory usage.
    """
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
        try:
            pinfo = proc.info
            # Safety Check for macOS/Windows
            if pinfo['memory_info'] is None:
                continue

            mem_gb = pinfo['memory_info'].rss / (1024 ** 3)
            
            processes.append({
                "PID": pinfo['pid'],
                "Name": pinfo['name'],
                "Memory (GB)": round(mem_gb, 2)
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    if not processes:
        return pd.DataFrame(columns=["PID", "Name", "Memory (GB)"])

    df = pd.DataFrame(processes)
    df = df.sort_values(by="Memory (GB)", ascending=False).head(20) # Top 20
    return df

def show_system_monitor():
    """
    Displays a system monitor with Multiselect Killer.
    """
    with st.sidebar.expander("💻 System Performance", expanded=False):
        
        # --- 1. VITALS ---
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        
        col1, col2 = st.columns(2)
        col1.metric("CPU", f"{cpu}%")
        col2.metric("RAM", f"{ram.percent}%")
        
        # RAM Bar Color
        if ram.percent < 70:
            st.progress(ram.percent / 100)
        else:
            st.warning("⚠️ High Memory Usage")
            st.progress(ram.percent / 100)

        st.markdown("---")

        # --- 2. THE PROCESS TABLE (View Only) ---
        st.write("### 🕵️ Memory Hogs")
        if st.button("🔄 Refresh List"):
            st.rerun()

        df = get_process_df()
        
        # Show clean table
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "PID": st.column_config.NumberColumn("PID", format="%d"),
                "Memory (GB)": st.column_config.NumberColumn("Mem", format="%.2f GB"),
            }
        )

        st.markdown("---")

        # --- 3. THE KILL SWITCH (Multiselect) ---
        st.write("### 🔫 Task Manager")
        
        # Create a list of strings like: "python (PID: 12345)"
        # We use a dictionary to map the string back to the PID
        options = {}
        for index, row in df.iterrows():
            label = f"{row['Name']} ({int(row['PID'])}) - {row['Memory (GB)']} GB"
            options[label] = int(row['PID'])

        # Multi-select dropdown
        selected_labels = st.multiselect(
            "Select processes to terminate:",
            options=list(options.keys()),
            placeholder="Choose processes..."
        )

        # BUTTON LOGIC: Only show if something is selected
        if selected_labels:
            st.write(f"⚠️ About to kill {len(selected_labels)} process(es)")
            
            # The Button
            if st.button("💀 Kill Selected Now", type="primary"):
                for label in selected_labels:
                    pid = options[label]
                    try:
                        p = psutil.Process(pid)
                        p.terminate()
                        st.toast(f"✅ Killed PID {pid}")
                    except Exception as e:
                        st.error(f"❌ Could not kill PID {pid}: {e}")
                
                time.sleep(1)
                st.rerun()

        st.markdown("---")
        if st.button("🔴 Stop App (Emergency)"):
            os._exit(0)