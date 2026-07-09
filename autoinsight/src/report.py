import json
import plotly.express as px
import plotly.io as pio
from jinja2 import Template
import os

def generate_html_report(df, cleaning_log, eda_profile, llm_narrative, output_path="reports/autoinsight_report.html"):
    """
    Generates a standalone HTML report containing LLM narrative, cleaning logs, and Plotly charts.
    """
    numeric_cols = df.select_dtypes(include=['number']).columns
    charts_html = ""
    
    if len(numeric_cols) >= 2:
        # Correlation heatmap
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, title="Correlation Matrix", color_continuous_scale='RdBu_r')
        charts_html += pio.to_html(fig_corr, full_html=False, include_plotlyjs='cdn')
        
    if len(numeric_cols) >= 1:
        # First numeric col distribution
        fig_dist = px.histogram(df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}", template="plotly_white")
        charts_html += pio.to_html(fig_dist, full_html=False, include_plotlyjs=False)

    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>AutoInsight Report</title>
        <style>
            body { font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px auto; max-width: 1200px; color: #333; line-height: 1.6; }
            h1, h2, h3 { color: #2c3e50; }
            .header { border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 30px; }
            .section { margin-bottom: 40px; padding: 25px; border: 1px solid #e1e8ed; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
            .llm-insights { background-color: #f8fbff; border-left: 5px solid #3498db; }
            pre { background: #f4f6f8; padding: 15px; overflow-x: auto; border-radius: 5px; font-family: 'Courier New', Courier, monospace; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>AutoInsight Data Report</h1>
            <p>Automated Exploratory Data Analysis & Cleaning Summary</p>
        </div>
        
        <div class="section llm-insights">
            <h2>LLM Insights & Narrative</h2>
            <div>{{ llm_narrative }}</div>
        </div>
        
        <div class="section">
            <h2>Cleaning Log</h2>
            <pre>{{ cleaning_log }}</pre>
        </div>
        
        <div class="section">
            <h2>Interactive Charts</h2>
            {{ charts_html }}
        </div>
    </body>
    </html>
    """
    
    # We replace newline characters for simple formatting, though markdown conversion would be ideal.
    formatted_narrative = llm_narrative.replace('\\n', '<br>')
    
    template = Template(template_str)
    html_content = template.render(
        cleaning_log=json.dumps(cleaning_log, indent=2),
        llm_narrative=formatted_narrative,
        charts_html=charts_html
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"Report successfully generated at {output_path}")
