import graphviz
import os

def create_hpg_diagram():
    print("Generating HPG Model Architecture diagram...")
    
    dot = graphviz.Digraph('HPG_Pipeline')

    # Base attributes
    dot.attr(dpi='1200', nodesep='0.8', ranksep='0.6', fontname='Helvetica', compound='true')
    dot.attr('node', shape='box', style='filled,rounded', fontname='Helvetica', fontsize='12', margin='0.25')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # Color palette
    colors = {
        'clean': '#F3E5F5',    # Light Purple
        'filter': '#E8F5E9',   # Light Green
        'validate': '#E0F2F1', # Teal
        'model': '#E8EAF6',    # Indigo
        'loss': '#FFEBEE'      # Light Red
    }

    # ==========================================
    # HPG MODEL ARCHITECTURE
    # ==========================================
    with dot.subgraph(name='cluster_model') as c2:
        c2.attr(label='HPG MODEL ARCHITECTURE', fontname='Helvetica-Bold', fontsize='14', style='dashed', margin='20')

        c2.node('enc', 'ENCODER (Base Model)\nallenai/PRIMERA\nInput: [article_tokens, mask]\nOutput: token_states (B, T, H)', fillcolor=colors['model'])

        c2.node('plan', 'SALIENCE-AWARE PLANNER\n- Segment Pooler (16 fixed segments)\n- Segment Attention Layer\n- Salience Head (NN scoring)\n- Query Attention (6 learnable queries)\n- Plan Refiner (2-layer Transformer)\nOutputs: plan_tokens (B, 6, H) + scores', fillcolor=colors['validate'])
        
        c2.node('fuse', 'PLAN-CONDITIONED FUSION\n- Token-to-Plan Attention\n- Gated Addition (Sigmoid)\n- FFN + LayerNorm\nOutputs: fused_tokens (B, T, H)', fillcolor=colors['clean'])

        c2.node('concat', 'CONCATENATION & MASKING\nState: [plan_tokens || fused_tokens]\nMask: [plan_mask || orig_mask]\nConditioned Output: (B, T+6, H)', fillcolor='#F5F5F5')

        c2.node('dec', 'DECODER + GENERATION\nSeq2Seq Decoder\nAutoregressive summary_tokens generation', fillcolor=colors['model'])
        
        c2.node('loss', 'MULTI-TASK LOSS\n- NLLLoss (Main Task)\n- Planner Entropy Loss (w=0.01)\n- Redundancy Loss (w=0.03)', shape='note', fillcolor=colors['loss'])
        
        c2.node('out2', 'FINAL OUTPUT\n- Generated Summary (Primary)\n- Plan Tokens (Interpretable)\n- Salience Scores', fillcolor=colors['filter'], shape='folder')

        c2.edge('enc', 'plan')
        c2.edge('enc', 'fuse')
        c2.edge('plan', 'fuse', label=' queries context')
        c2.edge('plan', 'concat')
        c2.edge('fuse', 'concat')
        c2.edge('concat', 'dec')
        c2.edge('dec', 'loss', style='dashed')
        c2.edge('dec', 'out2')
        
        # Keep the loss and output boxes side-by-side
        c2.body.append('\t{ rank=same; "loss"; "out2" }')
        
    # NOTE: The "dot.edge('out1', 'enc'...)" line that caused the blue inference flow has been completely removed.

    # ==========================================
    # RENDER FILES
    # ==========================================
    try:
        dot.render('hpg_architecture', format='ps', cleanup=True)
        dot.render('hpg_architecture', format='jpg', cleanup=True)
        
        print("Success!")
        print(f"- PostScript saved to: {os.path.abspath('hpg_architecture.ps')}")
        print(f"- JPEG saved to: {os.path.abspath('hpg_architecture.jpg')}")
        
    except Exception as e:
        print(f"Error rendering diagram: {e}")

if __name__ == '__main__':
    create_hpg_diagram()