# Regenerate everything the manuscript includes, from measurement to figure.
#   make ch12   - rerun Chapter 12's experiment, figures and tables
#   make lock   - refresh environment.lock
.PHONY: ch13 ch12 figures tables manuscript check xref lock clean

export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
export MKL_NUM_THREADS ?= 4
export PYTHONHASHSEED ?= 0

ch13:
	python3 -m bench.run_ch13
	$(MAKE) figures tables manuscript

ch12:
	python3 -m bench.run_ch12
	$(MAKE) figures tables manuscript

figures:
	python3 -m bench.figures

tables:
	python3 -m bench.tables

manuscript:
	python3 -m bench.manuscript

check: xref
	python3 -m bench.manuscript --check

xref:
	python3 -m bench.xref

lock:
	python3 -m pip freeze > environment.lock
	@python3 -c "import sys,platform,pathlib; \
p=pathlib.Path('environment.lock'); \
p.write_text('# python %s on %s\n%s' % (sys.version.split()[0], platform.platform(), p.read_text()))"

clean:
	rm -rf figures tables results/*.json __pycache__ */__pycache__
