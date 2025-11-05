import streamlit as st
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from app.monitoring.metrics import metrics_db
from app.orchestrator.chains import orchestrator
from app.core.vector_store import vector_store
import yaml

st.set_page_config(page_title="AI Orchestrator Dashboard", layout="wide", page_icon="🤖")

API_URL = "http://localhost:8000/api/v1"

st.title(" AI Orchestrator & Monitoring Platform")
st.markdown("---")

tabs = st.tabs([" Home", " Orchestrator", " Monitoring", " MCP Server", " Prompts", " Settings"])

with tabs[0]:
    st.header("Welcome to AI Orchestrator Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents in DB", vector_store.get_stats()["total_vectors"])
    
    with col2:
        metrics = metrics_db.get_metrics(limit=1000)
        st.metric("Total Queries", len(metrics))
    
    with col3:
        avg_latencies = metrics_db.get_avg_latency_by_tool()
        if avg_latencies:
            avg_lat = sum([lat for _, lat in avg_latencies]) / len(avg_latencies)
            st.metric("Avg Latency", f"{avg_lat:.2f}s")
        else:
            st.metric("Avg Latency", "N/A")
    
    st.markdown("### System Overview")

with tabs[1]:
    st.header(" AI Orchestrator")
    
    st.subheader("Upload Document")
    doc_text = st.text_area("Document Content", height=150)
    doc_topic = st.text_input("Topic/Category")
    
    if st.button(" Upload Document"):
        if doc_text:
            response = requests.post(
                f"{API_URL}/documents",
                json={"content": doc_text, "metadata": {"topic": doc_topic}}
            )
            if response.status_code == 200:
                st.success(f" Document uploaded! {response.json()['chunks_created']} chunks created")
            else:
                st.error(f" Error: {response.text}")
    
    st.markdown("---")
    st.subheader("Orchestrated Query")
    
    query = st.text_input("Enter your query")
    
    col1, col2, col3,col4 = st.columns(4)
    use_rag = col1.checkbox(" RAG", value=True)
    use_summarize = col2.checkbox(" Summarize")
    use_translate = col3.checkbox(" Translate")
    use_finetuned = col4.checkbox(" Fine-tuned")

    
    if st.button(" Execute Orchestration"):
        if query:
            tools = []
            if use_rag:
                tools.append("rag")
            if use_summarize:
                tools.append("summarize")
            if use_translate:
                tools.append("translate")
            if use_finetuned:
                tools.append("finetuned")

            
            with st.spinner("Processing..."):
                results = orchestrator.orchestrate(query, tools)
                
                for tool, result in results.items():
                    st.markdown(f"### {tool.upper()} Result")
                    if tool == "rag":
                        st.write(f"**Answer:** {result['answer']}")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        st.write(f"**Latency:** {result['latency']:.2f}s")
                        
                        metrics_db.log_metric(
                            tool="rag",
                            latency=result['latency'],
                            query=query,
                            result=result['answer'],
                            confidence=result['confidence']
                        )
                    
                    elif tool == "summarize":
                        st.write(f"**Summary:** {result['summary']}")
                        st.write(f"**Latency:** {result['latency']:.2f}s")
                        
                        metrics_db.log_metric(
                            tool="summarizer",
                            latency=result['latency'],
                            query=query,
                            result=result['summary']
                        )
                    
                    elif tool == "translate":
                        st.write(f"**Translation:** {result['translation']}")
                        st.write(f"**Latency:** {result['latency']:.2f}s")
                        
                        metrics_db.log_metric(
                            tool="translator",
                            latency=result['latency'],
                            query=query,
                            result=result['translation']
                        )
                    
                    st.markdown("---")

with tabs[2]:
    st.header(" Monitoring Dashboard")
    
    metrics = metrics_db.get_metrics(limit=100)
    
    if metrics:
        df = pd.DataFrame([
            {
                "timestamp": m.timestamp,
                "tool": m.tool,
                "latency": m.latency,
                "confidence": m.confidence
            }
            for m in metrics
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Latency by Tool")
            fig = px.box(df, x="tool", y="latency", color="tool")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Query Distribution")
            tool_counts = df['tool'].value_counts()
            fig = px.pie(values=tool_counts.values, names=tool_counts.index)
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Latency Over Time")
        fig = px.line(df, x="timestamp", y="latency", color="tool")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Recent Metrics")
        st.dataframe(df.head(20), use_container_width=True)
    else:
        st.info("No metrics available yet. Run some queries to see monitoring data!")

with tabs[3]:
    st.header(" MCP Server")
    
    st.markdown("""
    The Model Context Protocol (MCP) allows external applications to interact with your AI assistant.
    
    **Available Commands:**
    - `query-docs`: Query the RAG system
    - `summarize`: Summarize text
    - `translate`: Translate English to French
    """)
    
    st.subheader("Execute MCP Command")
    
    command = st.selectbox("Command", ["query-docs", "summarize", "translate"])
    
    if command == "query-docs":
        arg_value = st.text_input("Query")
        args = {"query": arg_value}
    elif command == "summarize":
        arg_value = st.text_area("Text to Summarize")
        args = {"text": arg_value}
    else:
        arg_value = st.text_area("Text to Translate")
        args = {"text": arg_value}
    
    if st.button("Execute MCP Command"):
        if arg_value:
            response = requests.post(
                f"{API_URL}/mcp/execute",
                json={"command": command, "args": args}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(" Command executed successfully!")
                st.json(result)
            else:
                st.error(f" Error: {response.text}")

with tabs[4]:
    st.header(" Prompt Registry")
    
    try:
        with open("app/prompts/registry.yaml", "r") as f:
            prompts = yaml.safe_load(f)
        
        for prompt_name, prompt_data in prompts["prompts"].items():
            with st.expander(f" {prompt_name} (v{prompt_data['version']})"):
                st.write(f"**Description:** {prompt_data['description']}")
                st.code(prompt_data['template'], language="text")
    except FileNotFoundError:
        st.warning("Prompt registry not found. Create app/prompts/registry.yaml")

with tabs[5]:
    st.header(" Settings")
    
    st.subheader("System Configuration")
    
    st.write("**Vector Database Stats:**")
    stats = vector_store.get_stats()
    st.json(stats)
    
    st.markdown("---")
    
    if st.button(" Clear All Documents"):
        vector_store.clear_index()
        st.success("All documents cleared!")
    
    if st.button(" Clear Metrics"):
        metrics_db.session.query(metrics_db.session.query(metrics_db.Metric).delete())
        metrics_db.session.commit()
        st.success("All metrics cleared!")