# Regenerate everything the manuscript includes, from measurement to figure.
#   make ch12   - rerun Chapter 12's experiment, figures and tables
#   make lock   - refresh environment.lock
.PHONY: all ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17 figures tables manuscript check xref audit terms tests lock clean

export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
export MKL_NUM_THREADS ?= 4
export PYTHONHASHSEED ?= 0

all: ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17

ch01:
	python3 -m bench.run_ch01
	$(MAKE) figures tables manuscript

ch02:
	python3 -m bench.run_ch02
	$(MAKE) figures tables manuscript

ch03:
	python3 -m bench.run_ch03
	$(MAKE) figures tables manuscript

ch04:
	python3 -m bench.run_ch04
	$(MAKE) figures tables manuscript

ch05:
	python3 -m bench.run_ch05
	$(MAKE) figures tables manuscript

ch06:
	python3 -m bench.run_ch06
	$(MAKE) figures tables manuscript

ch07:
	python3 -m bench.run_ch07
	$(MAKE) figures tables manuscript

ch08:
	python3 -m bench.run_ch08
	$(MAKE) figures tables manuscript

ch09:
	python3 -m bench.run_ch09
	$(MAKE) figures tables manuscript

ch10:
	python3 -m bench.run_ch10
	$(MAKE) figures tables manuscript

ch11:
	python3 -m bench.run_ch11
	$(MAKE) figures tables manuscript

ch14:
	python3 -m bench.run_ch14
	$(MAKE) figures tables manuscript

ch13:
	python3 -m bench.run_ch13
	$(MAKE) figures tables manuscript

ch12:
	python3 -m bench.run_ch12
	$(MAKE) figures tables manuscript

ch15:
	python3 -m bench.run_ch15
	$(MAKE) figures tables manuscript

ch16:
	python3 -m bench.run_ch16
	$(MAKE) figures tables manuscript

ch17:
	python3 -m bench.run_ch17
	$(MAKE) figures tables manuscript

figures:
	python3 -m bench.figures

tables:
	python3 -m bench.tables

manuscript:
	python3 -m bench.manuscript

check: xref tests audit terms
	python3 -m bench.manuscript --check

xref:
	python3 -m bench.xref

audit:
	python3 -m bench.audit

terms:
	python3 -m bench.terms

tests:
	@python3 -c "from bench.theme import test_theme_ramp_is_monotonic as a; a(); \
from tinyserve.reference import test_reference_is_an_8b_model as b; b(); \
from bench.listings import test_extract_is_exact as c; c(); \
from tinyserve.prefix import test_chained_hash_distinguishes_context as d; d(); \
from tinyserve.prefix import test_shared_blocks_outlive_their_sequence as e; e(); \
from tinyserve.prefix import test_structured_eviction_never_strands as f; f(); \
from tinyserve.batch import test_batching_changes_no_token as g; g(); \
from tinyserve.batch import test_decode_step_advances_every_sequence as h; h(); \
from tinyserve.scheduler import test_every_request_is_served_exactly_once as i; i(); \
from tinyserve.scheduler import test_continuous_finishes_short_requests_first as j; j(); \
from tinyserve.scheduler import test_a_small_pool_preempts_and_still_finishes_everyone as k; k(); \
from tinyserve.serving import test_prefill_is_compute_bound_and_decode_is_not as l; l(); \
print('tests: palette ramp, reference model is 8B, listings extract, prefix sharing, batched decode, scheduling')"

lock:
	python3 -m pip freeze > environment.lock
	@python3 -c "import sys,platform,pathlib; \
p=pathlib.Path('environment.lock'); \
p.write_text('# python %s on %s\n%s' % (sys.version.split()[0], platform.platform(), p.read_text()))"

clean:
	rm -rf figures tables results/*.json __pycache__ */__pycache__
