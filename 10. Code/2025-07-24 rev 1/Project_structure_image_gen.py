import matplotlib.pyplot as plt
import networkx as nx
from textwrap import wrap

# Set up the figure with a blueprint-like style
fig = plt.figure(figsize=(16, 10), facecolor='#E6F0FA')  # Larger size for better readability
plt.title('AI Trading Bot Blueprint', fontsize=18, pad=30, color='navy')

# Create a directed graph for module structure
G = nx.DiGraph()

# Add nodes with shortened labels for clarity
G.add_node("main.py\n(Entry Point)", level=1, color='lightblue')
G.add_node("config.py\n(Parameters)", level=2, color='lightgreen')
G.add_node("logging_utils.py\n(Logs Setup)", level=2, color='lightgreen')
G.add_node("trading_loop.py\n(Core Loop)", level=3, color='lightyellow')
G.add_node("data_utils.py\n(Data Fetch)", level=4, color='lightcoral')
G.add_node("prediction.py\n(Grok-3 Calls)", level=4, color='lightcoral')
G.add_node("order_execution.py\n(Orders/P/L)", level=4, color='lightcoral')
G.add_node("trade_analysis.py\n(P/L & Plots)", level=4, color='lightcoral')
G.add_node("historical_analysis_AAPL.json\n(Initial Data)", level=5, color='lightgray')

# Add edges for module dependencies
G.add_edge("main.py\n(Entry Point)", "config.py\n(Parameters)")
G.add_edge("main.py\n(Entry Point)", "logging_utils.py\n(Logs Setup)")
G.add_edge("trading_loop.py\n(Core Loop)", "main.py\n(Entry Point)")
G.add_edge("data_utils.py\n(Data Fetch)", "trading_loop.py\n(Core Loop)")
G.add_edge("prediction.py\n(Grok-3 Calls)", "trading_loop.py\n(Core Loop)")
G.add_edge("order_execution.py\n(Orders/P/L)", "trading_loop.py\n(Core Loop)")
G.add_edge("trade_analysis.py\n(P/L & Plots)", "trading_loop.py\n(Core Loop)")
G.add_edge("data_utils.py\n(Data Fetch)", "historical_analysis_AAPL.json\n(Initial Data)")

# Use a better layout for spacing
pos = nx.spring_layout(G, seed=42, k=0.5)  # Spring layout for auto-spacing

# Draw nodes with improved styling
node_colors = [G.nodes[n]['color'] for n in G.nodes]
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, edgecolors='black', linewidths=1.5, node_shape='o')
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Draw edges with arrows
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20, edge_color='gray', width=2)

# Process flow as a vertical list on the right
flow_steps = [
    "1. Start (main.py: Init Clients/Logging)",
    "2. Load Config (config.py: Modes/Keys/Params)",
    "3. Setup Logging (logging_utils.py: Configure File/Stream Handlers)",
    "4. Trading Loop (trading_loop.py: Session Check/Cooldown)",
    "5. Fetch Bars (data_utils.py: Load JSON Supports/Resistances + Sim or Real Data)",
    "6. Get Prediction (prediction.py: Grok-3 API Call with JSON Stats)",
    "7. If Signal: Execute Order (order_execution.py: Submit/Poll/Update Position)",
    "8. Analyze Trades (trade_analysis.py: P/L/Win Rate/Plot Equity)",
    "9. Sleep/Repeat or End Session (Close Positions if Needed)"
]
plt.text(1.1, 0.95, "Process Flow:", transform=plt.gca().transAxes, fontsize=12, fontweight='bold', color='darkblue')
for i, step in enumerate(flow_steps):
    wrapped_step = '\n'.join(wrap(step, width=40))  # Wrap text for readability
    plt.text(1.1, 0.9 - i * 0.08, wrapped_step, transform=plt.gca().transAxes, fontsize=9, color='darkblue', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Remove axes for clean blueprint look
plt.axis('off')

# Save the figure
plt.savefig('ai_trading_bot_blueprint_improved.png', dpi=300, bbox_inches='tight')
plt.close()

print("Improved blueprint diagram saved as 'ai_trading_bot_blueprint_improved.png' in the current directory.")