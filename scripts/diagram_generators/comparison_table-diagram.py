import graphviz
import os

def create_comparison_diagram():
    print("Generating Summarization Architecture Comparison diagram...")
    
    dot = graphviz.Digraph('Summarization_Comparison')

    # Base attributes - 300 DPI for high quality without crashing Windows
    dot.attr(dpi='1200', nodesep='0.6', ranksep='0.8', fontname='Helvetica', compound='true')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica', fontsize='12', margin='0.25')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Color palette
    colors = {
        'trad_blue': '#E3F2FD',   # Light Blue (Traditional)
        'long_green': '#E8F5E9',  # Light Green (Modern Baselines)
        'hpg_orange': '#FFF8E1',  # Light Yellow/Orange (HPG)
        'table_gray': '#F5F5F5',  # Light Gray
        'header_gray': '#E0E0E0'  # Darker Gray for headers
    }

    # SECTION 3: COMPARISON TABLE
    # ==========================================
    with dot.subgraph(name='cluster_s3') as s3:
        s3.attr(label='ARCHITECTURE FEATURE COMPARISON', fontname='Helvetica-Bold', fontsize='14', margin='20')

        # We construct an HTML-like label for Graphviz to draw a literal table
        table_html = f'''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">
            <TR>
                <TD BGCOLOR="{colors['header_gray']}"><B>Architecture / Model</B></TD>
                <TD BGCOLOR="{colors['header_gray']}"><B>Interpretability</B></TD>
                <TD BGCOLOR="{colors['header_gray']}"><B>Multi-Doc Support</B></TD>
                <TD BGCOLOR="{colors['header_gray']}"><B>Context Length</B></TD>
                <TD BGCOLOR="{colors['header_gray']}"><B>Computational Cost</B></TD>
                <TD BGCOLOR="{colors['header_gray']}"><B>Planning Transparency</B></TD>
                <TD BGCOLOR="{colors['header_gray']}"><B>Performance Status</B></TD>
            </TR>
            <TR>
                <TD BGCOLOR="{colors['trad_blue']}">Extractive Summarization</TD>
                <TD BGCOLOR="{colors['table_gray']}">High</TD>
                <TD BGCOLOR="{colors['table_gray']}">No (Basic)</TD>
                <TD BGCOLOR="{colors['table_gray']}">Short</TD>
                <TD BGCOLOR="{colors['table_gray']}">Low</TD>
                <TD BGCOLOR="{colors['table_gray']}">N/A</TD>
                <TD BGCOLOR="{colors['table_gray']}">Legacy Baseline</TD>
            </TR>
            <TR>
                <TD BGCOLOR="{colors['trad_blue']}">Abstractive (Seq2Seq)</TD>
                <TD BGCOLOR="{colors['table_gray']}">Low</TD>
                <TD BGCOLOR="{colors['table_gray']}">No</TD>
                <TD BGCOLOR="{colors['table_gray']}">Short (512 tokens)</TD>
                <TD BGCOLOR="{colors['table_gray']}">High</TD>
                <TD BGCOLOR="{colors['table_gray']}">Black-Box</TD>
                <TD BGCOLOR="{colors['table_gray']}">Standard Baseline</TD>
            </TR>
            <TR>
                <TD BGCOLOR="{colors['trad_blue']}">Hierarchical Attention</TD>
                <TD BGCOLOR="{colors['table_gray']}">Medium</TD>
                <TD BGCOLOR="{colors['table_gray']}">Limited</TD>
                <TD BGCOLOR="{colors['table_gray']}">Medium (1024 tokens)</TD>
                <TD BGCOLOR="{colors['table_gray']}">Very High</TD>
                <TD BGCOLOR="{colors['table_gray']}">Attention-Weight Only</TD>
                <TD BGCOLOR="{colors['table_gray']}">Historical SOTA</TD>
            </TR>
            <TR>
                <TD BGCOLOR="{colors['long_green']}">Long-Context (LED/PRIMERA)</TD>
                <TD BGCOLOR="{colors['table_gray']}">Low</TD>
                <TD BGCOLOR="{colors['table_gray']}">Yes (Concatenated)</TD>
                <TD BGCOLOR="{colors['table_gray']}">Long (4k - 16k tokens)</TD>
                <TD BGCOLOR="{colors['table_gray']}">Very High</TD>
                <TD BGCOLOR="{colors['table_gray']}">Black-Box</TD>
                <TD BGCOLOR="{colors['table_gray']}">Modern SOTA</TD>
            </TR>
            <TR>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>Hierarchical Planner-Generator</B></TD>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>High</B></TD>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>Yes (Explicit Selection)</B></TD>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>Long (4k+ tokens)</B></TD>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>High</B></TD>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>Fully Transparent</B></TD>
                <TD BGCOLOR="{colors['hpg_orange']}"><B>Proposed SOTA</B></TD>
            </TR>
        </TABLE>
        >'''
        
        # The table is assigned to a borderless node
        s3.node('table', table_html, shape='plaintext', fillcolor='none')

    # ==========================================
    # RENDER FILES
    # ==========================================
    try:
        dot.render('comparison_table', format='ps', cleanup=True)
        dot.render('comparison_table', format='jpg', cleanup=True)
        
        print("Success!")
        print(f"- PostScript saved to: {os.path.abspath('comparison_table.ps')}")
        print(f"- JPEG saved to: {os.path.abspath('comparison_table.jpg')}")
        
    except Exception as e:
        print(f"Error rendering diagram: {e}")

if __name__ == '__main__':
    create_comparison_diagram()