# Regenerate everything the manuscript includes, from measurement to figure.
#   make ch12   - rerun Chapter 12's experiment, figures and tables
#   make lock   - refresh environment.lock
.PHONY: all ch01 ch02 ch03 ch13 ch12 figures tables manuscript check xref theme lock clean

export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
export MKL_NUM_THREADS ?= 4
export PYTHONHASHSEED ?= 0

all: ch01 ch02 ch03 ch13 ch12

ch01:
	python3 -m bench.run_ch01
	$(MAKE) figures tables manuscript

ch02:
	python3 -m bench.run_ch02
	$(MAKE) figures tables manuscript

ch03:
	python3 -m bench.run_ch03
	$(MAKE) figures tables manuscript

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

check: xref theme
	python3 -m bench.manuscript --check

xref:
	python3 -m bench.xref

theme:
	@python3 -c "from bench.theme import test_theme_ramp_is_monotonic as t; \
t(); print('theme: sequential ramp is monotonic in lightness')"

lock:
	python3 -m pip freeze > environment.lock
	@python3 -c "import sys,platform,pathlib; \
p=pathlib.Path('environment.lock'); \
p.write_text('# python %s on %s\n%s' % (sys.version.split()[0], platform.platform(), p.read_text()))"

clean:
	rm -rf figures tables results/*.json __pycache__ */__pycache__
