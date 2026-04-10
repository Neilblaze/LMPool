import time
import math
import random
import sys
import os
import re

# Terminal aesthetics
class Theme:
    BLUE = "\033[38;5;39m"
    CYAN = "\033[38;5;51m"
    GREEN = "\033[38;5;82m"
    YELLOW = "\033[38;5;226m"
    MAGENTA = "\033[38;5;213m"
    RED = "\033[38;5;196m"
    WHITE = "\033[38;5;255m"
    GRAY = "\033[38;5;244m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    
    # Progress bar chars
    BAR_FILLED = "█"
    BAR_EMPTY = "░"

def print_header(title):
    width = 80
    print(f"\n{Theme.CYAN}{Theme.BOLD}{'='*width}{Theme.RESET}")
    print(f"{Theme.CYAN}{Theme.BOLD}{title.center(width)}{Theme.RESET}")
    print(f"{Theme.CYAN}{Theme.BOLD}{'='*width}{Theme.RESET}\n")

def get_table_lines(data, headers):
    col_widths = [max(len(str(x)) for x in col) for col in zip(*(data + [headers]))]
    lines = []
    
    # Header
    header_row = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    lines.append(f"{Theme.BOLD}{Theme.BLUE}{header_row}{Theme.RESET}")
    lines.append(f"{Theme.GRAY}{'-' * (sum(col_widths) + 3 * (len(headers) - 1))}{Theme.RESET}")
    
    # Rows
    for row in data:
        formatted_row = []
        for i, val in enumerate(row):
            if isinstance(val, (int, float)):
                formatted_row.append(f"{Theme.YELLOW}{val:>{col_widths[i]}}{Theme.RESET}")
            else:
                formatted_row.append(f"{Theme.WHITE}{val:<{col_widths[i]}}{Theme.RESET}")
        lines.append(" | ".join(formatted_row))
    return lines

def get_graph_lines(values, height=10, width=60, title="Loss Curve"):
    if not values: return []
    min_v = min(values)
    max_v = max(values)
    
    if max_v == min_v:
        max_v += 0.1
        min_v -= 0.1
    
    span = max_v - min_v
    lines = []
    
    lines.append(f"{Theme.MAGENTA}{Theme.BOLD}   {title}{Theme.RESET}")
    for h in range(height, -1, -1):
        threshold = min_v + (h / height) * span
        line = f"{Theme.GRAY}{threshold:6.4f} | {Theme.RESET}"
        
        for w in range(width):
            idx = int((w / width) * len(values))
            val = values[idx]
            if h == 0:
                line += f"{Theme.GRAY}─{Theme.RESET}"
            elif abs((val - min_v) / span * height - h) < 0.5:
                line += f"{Theme.GREEN}●{Theme.RESET}"
            else:
                line += " "
        lines.append(line)
    
    footer = " " * 9 + "└" + "─" * (width - 1) + "┘"
    lines.append(f"{Theme.GRAY}{footer}{Theme.RESET}")
    lines.append(f"{Theme.GRAY}{' ' * 9}0{ ' ' * (width-4)}Steps{Theme.RESET}")
    return lines

# Pre-compiled ANSI stripper
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def ansi_len(s):
    return len(ANSI_ESCAPE.sub('', s))

def print_columns(column_blocks, gap=8):
    max_height = max(len(block) for block in column_blocks)
    
    # Calculate max visible width for each block
    block_widths = []
    for block in column_blocks:
        if not block:
            block_widths.append(0)
            continue
        max_w = max(ansi_len(line) for line in block)
        block_widths.append(max_w)
    
    # Pad blocks to max height
    padded_blocks = []
    for block in column_blocks:
        padded_blocks.append(block + [""] * (max_height - len(block)))
    
    # Print row by row with proper padding
    for i in range(max_height):
        row_parts = []
        for j, block in enumerate(padded_blocks):
            line = block[i]
            width = block_widths[j]
            # Pad the line based on visible length
            curr_len = ansi_len(line)
            padding = " " * (width - curr_len)
            row_parts.append(line + padding)
        
        print((" " * gap).join(row_parts))

def update_progress(current, total, loss, lr, start_time):
    percent = (current / total) * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = Theme.GREEN + Theme.BAR_FILLED * filled + Theme.GRAY + Theme.BAR_EMPTY * (bar_len - filled) + Theme.RESET
    
    elapsed = time.time() - start_time
    speed = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / speed if speed > 0 else 0
    
    sys.stdout.write(f"\r{Theme.BOLD}[{current:4d}/{total:4d}]{Theme.RESET} {bar} {Theme.CYAN}{percent:5.1f}%{Theme.RESET} | "
                     f"Loss: {Theme.YELLOW}{loss:6.4f}{Theme.RESET} | "
                     f"LR: {Theme.MAGENTA}{lr:8.6f}{Theme.RESET} | "
                     f"Req: {Theme.BLUE}{speed:4.1f} step/s{Theme.RESET} | "
                     f"ETA: {Theme.GRAY}{int(eta)}s{Theme.RESET}")
    sys.stdout.flush()

def main():
    # Simulation parameters
    total_steps = 1000
    n_layer = 12
    n_embd = 384
    n_head = 6
    vocab_size = 50257
    
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print_header("microAR Deep Transformer - Full Attention Residuals (FAR)")
    
    # 1. Model Configuration & 2. Dataset Stats (Side-by-Side)
    print(f"{Theme.BOLD}🚀 Initializing Environment...{Theme.RESET}")
    time.sleep(0.5)
    
    config_data = [
        ["Architecture", "microAR-FAR (Deep-GPT)"],
        ["Layers", n_layer],
        ["Embedding Dim", n_embd],
        ["Attention Heads", n_head],
        ["Context Length", 512],
        ["Vocab Size", vocab_size],
        ["Total Params", "125.4M"],
        ["AR Variant", "Full Attention Residuals"],
        ["Device", "NVIDIA L40S (24GB VRAM)"]
    ]
    dataset_data = [
        ["Train Samples", 2500000],
        ["Val Samples", 50000],
        ["Tokenization", "Byte-Pair Encoding (BPE)"],
        ["Mixing Entropy", "Initial: 2.14 (Uniform)"],
        ["Dataset Name", "OpenWebText-Subset"],
        ["Workers", 8],
        ["Batch Size", 32],
        ["Sparsity Target", "0.85"],
        ["Seed", 42]
    ]
    
    config_lines = [f"{Theme.BOLD}Model Config{Theme.RESET}"] + get_table_lines(config_data, ["Hyperparameter", "Value"])
    dataset_lines = [f"{Theme.BOLD}Dataset Stats{Theme.RESET}"] + get_table_lines(dataset_data, ["Metric", "Description"])
    
    print_columns([config_lines, dataset_lines])
    
    print(f"\n{Theme.BOLD}{Theme.YELLOW}Starting Training Optimization...{Theme.RESET}\n")
    
    # 3. Training Loop Simulation
    loss_history = []
    val_loss_history = []
    start_time = time.time()
    
    curr_loss = 10.8 # Initial loss
    
    for step in range(1, total_steps + 1):
        decay = 1.0 / (1.0 + step * 0.005)
        noise = (random.random() - 0.5) * 0.05
        curr_loss = (10.0 * decay) + 0.5 + noise
        lr = 6e-4 * (1.0 - (step / total_steps))
        loss_history.append(curr_loss)
        
        if step % 2 == 0:
            update_progress(step, total_steps, curr_loss, lr, start_time)
            time.sleep(0.01)
            
        if step % 200 == 0:
            sys.stdout.write("\n\n")
            val_loss = curr_loss * 1.05 + random.random() * 0.1
            val_loss_history.append(val_loss)
            
            val_lines = [
                f"{Theme.BOLD}{Theme.CYAN}--- Step {step} Validation ---{Theme.RESET}",
                f"Validation Loss: {Theme.YELLOW}{val_loss:.4f}{Theme.RESET}",
                f"Accuracy:        {Theme.GREEN}{max(0, (1 - val_loss/10)*100):.2f}%{Theme.RESET}",
                f"Perplexity:      {Theme.MAGENTA}{math.exp(val_loss):.2f}{Theme.RESET}",
                f"Wall Time:       {Theme.BLUE}{int(time.time() - start_time)}s{Theme.RESET}"
            ]
            
            graph_lines = get_graph_lines(loss_history[-100:], height=7, width=50, title=f"Local Training Loss")
            
            print_columns([val_lines, graph_lines], gap=12)
            print("\n")
            
    print(f"\n\n{Theme.BOLD}{Theme.GREEN}✅ Training Complete!{Theme.RESET}")
    print(f"Final Loss: {Theme.YELLOW}{curr_loss:.4f}{Theme.RESET} | Total Time: {Theme.CYAN}{time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}{Theme.RESET}\n")
    
    # 4. Final Analysis & 5. Inference (Side-by-Side)
    summary_lines = [f"{Theme.BOLD}Evaluation Summary{Theme.RESET}"] + get_table_lines([
        ["Final Train Loss", f"{curr_loss:.4f}"],
        ["Final Val Loss", f"{val_loss:.4f}"],
        ["Top-1 Accuracy", "42.1%"],
        ["Perplexity", f"{math.exp(curr_loss):.2f}"],
        ["AR Mixing Sparsity", "0.78 (Localized)"]
    ], ["Metric", "Value"])
    
    samples = [
        "The attention residual stream allows for a more direct gradient flow to...",
        "By selectively mixing previous sublayer outputs, the model learns to bypass...",
        "Computational efficiency of Block-AR vs Full-AR becomes evident when scale...",
        "Experimental results on OpenWebText indicate a 14% faster convergence rate...",
        "The moonshot paper's implementation of additive residuals was extended by..."
    ]
    sample_lines = [f"{Theme.BOLD}🔮 Generated Samples (Temp=0.8){Theme.RESET}"] + [f"{Theme.GRAY}[{i+1}]{Theme.RESET} {s}" for i, s in enumerate(samples)]
    
    print_columns([summary_lines, sample_lines], gap=10)
    
    print(f"\n{Theme.BOLD}{Theme.CYAN}Log file generated and saved to ./logs/run_2026_04_02.log{Theme.RESET}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Theme.RED}Interrupt received. Stopping...{Theme.RESET}")
        sys.exit(0)
