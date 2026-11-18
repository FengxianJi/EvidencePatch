PYTHON ?= python3
.PHONY: reproduce figures demo verify
reproduce:
	$(PYTHON) scripts/reproduce_results.py
figures:
	$(PYTHON) scripts/build_feedback_figures.py
demo:
	$(PYTHON) scripts/run_method.py
verify:
	$(PYTHON) scripts/verify_release.py
