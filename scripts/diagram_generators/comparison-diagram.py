import graphviz
import os

def create_comparison_diagram():
    print("Generating Summarization Architecture Comparison diagram...")
    
    dot = graphviz.Digraph('Summarization_Comparison')

    # Base attributes - 1200 DPI for high quality without crashing Windows
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

    # ==========================================
    # SECTION 1: TRADITIONAL TECHNIQUES
    # ==========================================
    with dot.subgraph(name='cluster_s1') as s1:
        s1.attr(label='TRADITIONAL & BASELINE TECHNIQUES', fontname='Helvetica-Bold', fontsize='14', style='dashed', margin='20')

        # Note: Using \l instead of \n to explicitly left-align the bullet points
        ext_text = (
            "1. Extractive Summarization\l\l"
            "- Key: Select important sentences/passages\l"
            "- Method: TF-IDF, TextRank, SumBasic\l"
            "- Pros: Preserves original text, very fast\l"
            "- Cons: No paraphrasing, limited coherence\l"
        )
        s1.node('ext', ext_text, fillcolor=colors['trad_blue'])

        abs_text = (
            "2. Abstractive Summarization (Seq2Seq)\l\l"
            "- Key: Generate new text from scratch\l"
            "- Method: Encoder-Decoder, RNNs, LSTMs\l"
            "- Pros: Coherent, concise output\l"
            "- Cons: Computationally expensive, hallucination\l"
        )
        s1.node('abs', abs_text, fillcolor=colors['trad_blue'])

        hier_text = (
            "3. Hierarchical Attention (Trad)\l\l"
            "- Key: Document + Sentence level attention\l"
            "- Method: Multi-level RNNs/Transformers\l"
            "- Pros: Captures document structure\l"
            "- Cons: Limited context window, complex training\l"
        )
        s1.node('hier', hier_text, fillcolor=colors['trad_blue'])

        long_text = (
            "4. Long-Context Models (Modern)\l\l"
            "- Key: Extended context with sparse attention\l"
            "- Method: LED, Longformer, LongT5, PRIMERA\l"
            "- Pros: Handles much longer inputs\l"
            "- Cons: Flat attention, less interpretable planning\l"
        )
        s1.node('long', long_text, fillcolor=colors['long_green'])

        # Force these 4 nodes to align side-by-side horizontally
        s1.body.append('\t{ rank=same; "ext"; "abs"; "hier"; "long" }')
        
        # Invisible edges to enforce Left-to-Right order
        s1.edge('ext', 'abs', style='invis')
        s1.edge('abs', 'hier', style='invis')
        s1.edge('hier', 'long', style='invis')

    # ==========================================
    # SECTION 2: HIERARCHICAL PLANNER-GENERATOR
    # ==========================================
    with dot.subgraph(name='cluster_s2') as s2:
        s2.attr(label='HIERARCHICAL PLANNER-GENERATOR (HPG)', fontname='Helvetica-Bold', fontsize='14', style='dashed', margin='20')

        hpg_text = (
            "PROPOSED: Hierarchical Planner-Generator (HPG) Architecture\l\l"
            "How HPG Improves on Baselines:\l"
            " [+] Interpretable Planning: Explicit salience-aware segment planning before generation.\l"
            " [+] Hierarchical Structure: Segment pooling -> Plan refinement -> Fusion into Generation.\l"
            " [+] Control Mechanisms: Redundancy penalties & entropy regularization ensure diverse coverage.\l"
            " [+] Long-Context & Multi-Doc: Naturally handles multi-document inputs with explicit content selection.\l"
        )
        s2.node('hpg', hpg_text, fillcolor=colors['hpg_orange'], shape='box', penwidth='2.0', color='#FF8F00')

    # Connect Section 1 to Section 2
    dot.edge('abs', 'hpg', style='invis') # Hidden structural edge
    dot.edge('hier', 'hpg', label=' Evolves structure  ', fontname='Helvetica-Oblique', color='#546E7A')
    dot.edge('long', 'hpg', label=' Adds interpretability  ', fontname='Helvetica-Oblique', color='#546E7A')


    # ==========================================
    # RENDER FILES
    # ==========================================
    try:
        dot.render('architecture_comparison2', format='ps', cleanup=True)
        dot.render('architecture_comparison2', format='jpg', cleanup=True)
        
        print("Success!")
        print(f"- PostScript saved to: {os.path.abspath('architecture_comparison2.ps')}")
        print(f"- JPEG saved to: {os.path.abspath('architecture_comparison2.jpg')}")
        
    except Exception as e:
        print(f"Error rendering diagram: {e}")

if __name__ == '__main__':
    create_comparison_diagram()