# DeepSeek V3 Web Chat Interface

## Features

### 🟠 1. Dynamic Model Switching
- Automatically discovers all `.pt` files in the `checkpoints/` directory
- Models grouped by type (Pretrained, SFT, RL)
- Loading animation displayed during switching
- Real-time model loading status query
- Model caching mechanism for faster switching of already-loaded models
- Automatically cancels ongoing generation when switching models

### 🟠 2. Streaming Output
- Implemented using Server-Sent Events (SSE)
- Each token displayed in real-time in the chat box
- Supports temperature, top_p, top_k, repetition_penalty parameters
- Configurable max_new_tokens

### 🟠 3. Cancel Generation
- Cancel button displayed during generation
- Click cancel to immediately stop generation
- Automatically cancels current generation when switching models

### 🟠 4. ChatGPT-Style UI
- Dark theme design
- Responsive layout
- Auto-adjusting input box height
- Message bubbles displaying user and AI conversation
- Clear conversation function
- Error notification system
- Sidebar parameter adjustment

---

## Launch Methods

### Method 1: Using run.sh script
```bash
./scripts/run.sh web-chat
```

### Method 2: Direct Python execution
```bash
python3 chat/app.py
```

> [!NOTE]
> After service starts, visit: http://localhost:5001

## File Structure

```
chat/
├── app.py              # Flask backend main program
├── templates/
│   └── index.html      # Main page template
├── static/
│   ├── style.css       # Stylesheet
│   └── app.js          # Frontend JavaScript
└── RUN.md              # This document
```

## API Endpoints

### GET /api/checkpoints
Get list of all available checkpoints

### POST /api/load_model
Load specified model
```json
{"checkpoint_path": "/path/to/model.pt"}
```

### GET /api/model_status
Get current model status

### POST /api/generate
Stream text generation (SSE)
```json
{
    "prompt": "Hello",
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50
}
```

### POST /api/cancel
Cancel ongoing generation

---

## Important Notes

> [!IMPORTANT]
> **CPU Mode**: Web server runs on CPU to avoid MPS threading issues

> [!TIP]
> **Model Configuration**: Ensure checkpoint matches config_default.yaml configuration

> [!WARNING]
> **Memory Management**: Large models require sufficient memory

> [!NOTE]
> **Browser Support**: Requires modern browser with SSE support

## Launch Command

```bash
./scripts/run.sh web-chat
```
