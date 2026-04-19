PYTHON ?= python

.PHONY: help metrics tables core nvdcf

help:
	@echo "Available targets:"
	@echo "  make metrics    - print compact reference metrics from bundled artifacts"
	@echo "  make tables     - regenerate compact markdown tables from bundled artifacts"
	@echo "  make core       - run the core R1-R5 experiment block"
	@echo "  make nvdcf      - run the NvDCF export + comparison path"

metrics:
	$(PYTHON) scripts/extract_reference_metrics.py

tables:
	$(PYTHON) scripts/extract_reference_tables.py

core:
	bash scripts/run_core_experiments.sh

nvdcf:
	bash scripts/run_nvdcf_comparison.sh
