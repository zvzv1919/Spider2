.PHONY: install clean

install: .venv
	uv sync
	@echo "✓ Installation complete"
	@echo ""
	@echo "To activate the virtual environment, run:"
	@echo "  source .venv/bin/activate"

.venv:
	uv venv
	@echo "✓ Virtual environment created"

clean:
	rm -rf .venv dist/ build/ *.egg-info .ruff_cache .mypy_cache .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned"
