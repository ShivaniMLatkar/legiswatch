.PHONY: help install lint fmt typecheck test eval run review walkthrough clean

help:            ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:         ## Install the package with development extras
	pip install -e ".[dev,service,connectors]"

lint:            ## Lint and check formatting
	ruff check src tests
	ruff format --check src tests

fmt:             ## Apply formatting and safe lint fixes
	ruff format src tests
	ruff check --fix src tests

typecheck:       ## Run mypy
	mypy

test:            ## Run the test suite (hermetic; no network)
	pytest --cov --cov-report=term-missing

eval:            ## Score the pipeline against the gold set
	legiswatch-eval

run:             ## Run the pipeline and build the dashboard
	legiswatch-run

review:          ## List obligations awaiting review
	legiswatch-review list --route human_review

walkthrough:     ## Exercise the review layer end to end
	bash scripts/review_walkthrough.sh

clean:           ## Remove build and cache artefacts
	rm -rf out build dist .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
