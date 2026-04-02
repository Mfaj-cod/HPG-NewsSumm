import graphviz
import os

def create_dataprep_diagram():
    print("Generating Data Preparation Pipeline diagram...")
    
    dot = graphviz.Digraph('DataPrep_Pipeline')

    # Base attributes (1200 DPI for safe, high-quality rendering)
    dot.attr(dpi='800', nodesep='0.8', ranksep='0.6', fontname='Helvetica', compound='true')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica', fontsize='12', margin='0.25')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Color palette
    colors = {
        'data': '#E1F5FE',     # Light Blue
        'clean': '#F3E5F5',    # Light Purple
        'filter': '#E8F5E9',   # Light Green
        'dedup': '#FFF3E0',    # Light Orange
        'cluster': '#E0F7FA',  # Cyan
        'validate': '#E0F2F1', # Teal
        'output': '#E8EAF6'    # Indigo
    }

    # ==========================================
    # DATA PREPARATION PIPELINE
    # ==========================================
    with dot.subgraph(name='cluster_dataprep') as c1:
        c1.attr(label='DATA PREPARATION PIPELINE', fontname='Helvetica-Bold', fontsize='14', style='dashed', margin='20')

        c1.node('in', 'Raw News Articles (XLSX)\nINPUT', fillcolor=colors['data'], shape='note')
        c1.node('load', 'Data Loading & Resolution\nMapping: article_text, summary, headline, etc.', fillcolor=colors['data'])
        c1.node('clean', 'Text Cleaning\n- HTML Tag Removal (BeautifulSoup)\n- Punctuation Normalization\n- Boilerplate Filtering (Regex)', fillcolor=colors['clean'])
        c1.node('lang', 'Language Filtering\nLangDetect (Confidence > 80%)\nFiltered to English Only', fillcolor=colors['filter'])
        
        c1.node('dedup', 'Deduplication Layer\n1. Exact Hash (Fast Duplicate Elimination)\n2. MinHash (LSH, Sim > 0.9)\n3. TF-IDF (Cosine Sim > 0.95)', fillcolor=colors['dedup'])
        
        c1.node('cluster', 'Clustering Stage\nTF-IDF Vectorizer (max 50K features)\nSVD Reduction (256 components)\nEvaluated via Silhouette Score', fillcolor=colors['cluster'])
        
        c1.node('val', 'Summary Validation\nROUGE-L >= 0.1\nContent Overlap >= 0.2\nLength Ratio: 0.01 - 0.5x', fillcolor=colors['validate'])
        
        c1.node('out1', 'CLEAN DATASET\nnewssumm_enhanced.json\nOutputs: baseline_stats, enhanced_stats, logs', fillcolor=colors['output'], shape='folder')

        c1.edges([('in', 'load'), ('load', 'clean'), ('clean', 'lang'), 
                  ('lang', 'dedup'), ('dedup', 'cluster'), ('cluster', 'rebal'), 
                  ('rebal', 'val'), ('val', 'out1')])

    # ==========================================
    # RENDER FILES
    # ==========================================
    try:
        dot.render('dataprep_pipeline', format='ps', cleanup=True)
        dot.render('dataprep_pipeline', format='jpg', cleanup=True)
        
        print("Success!")
        print(f"- PostScript saved to: {os.path.abspath('dataprep_pipeline.ps')}")
        print(f"- JPEG saved to: {os.path.abspath('dataprep_pipeline.jpg')}")
        
    except Exception as e:
        print(f"Error rendering diagram: {e}")

if __name__ == '__main__':
    create_dataprep_diagram()