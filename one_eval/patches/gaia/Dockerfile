FROM python:3.11-slim

ENV HF_HOME=/root/.cache/huggingface
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

# Install smolagents + all GAIA dependencies (matching the setup commands from bench_gallery)
RUN uv pip install --system \
        'smolagents[openai]' \
        litellm \
        ddgs \
        markdownify \
        requests \
        datasets \
        huggingface-hub \
        tqdm \
        pandas \
        'pdfminer.six' \
        pypdf \
        python-pptx \
        mammoth \
        openpyxl \
        xlrd \
        puremagic \
        pathvalidate \
        beautifulsoup4 \
        Pillow \
        pydub \
        SpeechRecognition \
        youtube-transcript-api \
        google-search-results \
        sympy \
        pubchempy \
        biopython \
        scikit-learn \
        scipy \
        chess \
        numpy \
        numexpr \
    && rm -rf /root/.cache/pip /root/.cache/uv

WORKDIR /workspace/repo
