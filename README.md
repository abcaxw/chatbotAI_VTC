# RAG Multi-Agent Chatbot System

Hệ thống chatbot RAG (Retrieval-Augmented Generation) sử dụng kiến trúc multi-agent với LangGraph, chạy hoàn toàn local trên GPU với model GPT-OSS-20B và Vietnamese-SBERT embeddings.

## Kiến trúc hệ thống

### Multi-Agent Architecture
- **SUPERVISOR**: Điều phối chính, phân loại yêu cầu
- **FAQ**: Tìm kiếm câu hỏi thường gặp
- **RETRIEVER**: Tìm kiếm tài liệu từ vector database
- **GRADER**: Đánh giá chất lượng thông tin
- **GENERATOR**: Tạo câu trả lời từ thông tin đã lọc
- **NOT_ENOUGH_INFO**: Xử lý thiếu thông tin
- **CHATTER**: An ủi cảm xúc tiêu cực khách hàng
- **REPORTER**: Thông báo lỗi/bảo trì hệ thống
- **OTHER**: Xử lý yêu cầu ngoài phạm vi

### Tech Stack
- **Vector Database**: Milvus
- **Embedding Model**: keepitreal/vietnamese-sbert (768D)
- **LLM**: GPT-OSS-20B via Ollama
- **Framework**: LangGraph + FastAPI
- **Language**: Python 3.11+

## Cấu trúc dự án

```
Embedding_vectorDB/
├── config.py      # Cấu hình xử lý tài liệu
├── document_processor.py       # Module xử lý OCR file
├── embedding_service.py        # Module xử lý embeddiing nội dung văn bản
├── main.py    # file API chính
├── milvus_client.py # Khởi tạo vector database
RAG_Core/
├── agents/                 # Các agent xử lý
│   ├── supervisor.py      # Agent điều phối
│   ├── faq_agent.py       # Agent FAQ
│   ├── retriever_agent.py # Agent tìm kiếm
│   ├── grader_agent.py    # Agent đánh giá
│   ├── generator_agent.py # Agent tạo câu trả lời
│   ├── chatter_agent.py   # Agent xử lý cảm xúc
│   └── ... 
├── models/                # Models và embeddings
│   ├── embedding_model.py
│   └── llm_model.py
├── database/              # Kết nối Milvus
│   └── milvus_client.py
├── tools/                 # Function calling tools
│   └── vector_search.py
├── workflow/              # LangGraph workflow
│   └── rag_workflow.py
├── api/                   # FastAPI endpoints
│   ├── main.py
│   └── schemas.py
├── config/                # Cấu hình
│   └── settings.py
├── utils/                 # Utilities
│   └── helpers.py
├── docker-compose.yml     # Docker setup
├── requirements.txt       # Python dependencies
└── main.py               # Entry point
```

## Hướng dẫn cài đặt và chạy

### Yêu cầu hệ thống
- Python 3.11+
- Docker & Docker Compose
- GPU với ít nhất 16GB VRAM (để chạy GPT-OSS-20B)

### Bước 1: Clone và setup môi trường

```bash
# Tạo virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```


### Bước 2: Khởi động hệ thống với Docker

```bash
# Khởi động tất cả services (Milvus + Ollama + dependencies)
docker-compose up -d

# Kiểm tra trạng thái containers
docker-compose ps

# Xem logs nếu cần
docker-compose logs -f
```

### Bước 4: Setup Ollama và download model

```bash

# Tải vllm
pip install Ollama

# Download GPT-OSS-20B model
Ollama pull gpt-oss:20b

```

### Bước 5: Chạy API document process
```bash
# Chạy ứng dụng chính
cd Embedding_vectorDB
python main.py

# Hoặc chạy trực tiếp với uvicorn
uvicorn api.main:app --host localhost --port 8000 --reload

```
**document process sẽ chạy tại:** `http://localhost:8000`

### Bước 6: Chạy API RAG_core Multi Agent

```bash
# Chạy ứng dụng chính
cd RAG_Core
python main.py

# Hoặc chạy trực tiếp với uvicorn
uvicorn api.main:app --host localhost --port 8501 --reload
```

**chatbotAI sẽ chạy tại:** `http://localhost:8501`

### Bước 7: Test hệ thống

```bash
# Process Document
curl -X POST "http://localhost:8000/api/v1/process-document" \
  -F "file=@document.pdf"


# Test chat API (cần có data trong vector DB)
curl -X POST "http://localhost:8501/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "Làm thế nào để đăng ký dịch vụ?",
     }'
```

## Configuration

### Điều chỉnh Agent Behavior

Chỉnh sửa `config/settings.py`:
```python
# Similarity threshold cho vector search
SIMILARITY_THRESHOLD: float = 0.7  # Càng cao càng strict

# Số lượng documents trả về
TOP_K: int = 5

# Max iterations cho workflow  
MAX_ITERATIONS: int = 5
```

### Custom Prompts

Chỉnh sửa prompts trong các agent files:
- `agents/supervisor.py`: Prompt phân loại
- `agents/generator_agent.py`: Prompt tạo câu trả lời
- `agents/chatter_agent.py`: Prompt xử lý cảm xúc

---

**🎉 Chúc bạn thành công với RAG Multi-Agent Chatbot!**
