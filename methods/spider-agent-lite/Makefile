.PHONY: install sync run setup clean

# Install uv and sync dependencies for fresh environments
install:
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync

# Sync dependencies (assumes uv is already installed)
sync:
	uv sync

# Run the agent (requires OPENAI_API_KEY to be set)
run:
	uv run python run.py --model gpt-4o -s test1

# Setup Spider 2.0-Lite data
setup:
	uv run gdown 'https://drive.google.com/uc?id=1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG' -O ../../spider2-lite/resource/
	rm -rf ../../spider2-lite/resource/databases/spider2-localdb
	mkdir -p ../../spider2-lite/resource/databases/spider2-localdb
	unzip ../../spider2-lite/resource/local_sqlite.zip -d ../../spider2-lite/resource/databases/spider2-localdb
	uv run python spider_agent_setup_lite.py

# Clean up virtual environment and cache
clean:
	rm -rf .venv
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


