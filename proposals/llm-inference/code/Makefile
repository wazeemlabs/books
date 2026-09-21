# Regenerate everything the manuscript includes, from measurement to figure.
#   make ch12   - rerun Chapter 12's experiment, figures and tables
#   make ddr1   - re-derive a Part's design decision record from its chapters
#   make lock   - refresh environment.lock
.PHONY: all venv ddr1 ch28 ch30 ch32 ch33 ch22 ch24 ch29 ch31 ch41 ch42 ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17 ch18 ch19 ch20 ch22 ch24 ch29 ch31 ch41 ch42 figures tables manuscript check xref audit terms tests lock clean

export OMP_NUM_THREADS ?= 4
export OPENBLAS_NUM_THREADS ?= 4
export MKL_NUM_THREADS ?= 4
export PYTHONHASHSEED ?= 0

all: ch01 ch02 ch03 ch04 ch05 ch06 ch07 ch08 ch09 ch10 ch11 ch12 ch13 ch14 ch15 ch16 ch17 ch18 ch19 ch20 ch22 ch24 ch28 ch29 ch30 ch31 ch32 ch33 ch41 ch42 ddr1

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

ch18:
	python3 -m bench.run_ch18
	$(MAKE) figures tables manuscript

ch19:
	python3 -m bench.run_ch19
	$(MAKE) figures tables manuscript

ch20:
	python3 -m bench.run_ch20
	$(MAKE) figures tables manuscript

ch22:
	python3 -m bench.run_ch22
	$(MAKE) figures tables manuscript

ch24:
	python3 -m bench.run_ch24
	$(MAKE) figures tables manuscript

ch29:
	python3 -m bench.run_ch29
	$(MAKE) figures tables manuscript

ch41:
	python3 -m bench.run_ch41
	$(MAKE) figures tables manuscript

ch42:
	python3 -m bench.run_ch42
	$(MAKE) figures tables manuscript

ch28:
	python3 -m bench.run_ch28
	$(MAKE) figures tables manuscript

ch30:
	python3 -m bench.run_ch30
	$(MAKE) figures tables manuscript

ch33:
	python3 -m bench.run_ch33
	$(MAKE) figures tables manuscript

ch31:
	python3 -m bench.run_ch31
	$(MAKE) figures tables manuscript

ch32:
	python3 -m bench.run_ch32
	$(MAKE) figures tables manuscript


ddr1:
	python3 -m bench.run_ddr1
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
from tinyserve.serving import test_the_two_steps_are_one_step as m; m(); \
from tinyserve.scheduler import test_chunking_reads_every_prompt_token_exactly_once as n; n(); \
from tinyserve.scheduler import test_a_smaller_budget_smooths_the_gap_between_tokens as o; o(); \
from tinyserve.scheduler import test_swapping_keeps_the_tokens_that_recomputing_throws_away as q; q(); \
from tinyserve.scheduler import test_the_mean_batch_is_inside_the_batches_it_averages as r; r(); \
from tinyserve.flash import test_tiling_changes_no_number as s; s(); \
from tinyserve.flash import test_tiling_survives_every_tile_size as t; t(); \
from tinyserve.flash import test_the_byte_formulas_match_the_bytes_counted as u; u(); \
from tinyserve.flash import test_a_decode_step_skips_no_keys as v; v(); \
from tinyserve.flash import test_swapping_the_kernel_changes_no_token as w; w(); \
from tinyserve.flash import test_a_decode_step_has_almost_no_square_term_to_save as x; x(); \
from tinyserve.flash import test_keeping_the_block_resident_is_strictly_a_bound as y; y(); \
from tinyserve.device import test_the_accelerator_guard_refuses_what_it_should as z; z(); \
from tinyserve.device import test_the_contention_guard_reads_a_spread_the_load_average_cannot as z2; z2(); \
from tinyserve.precision import test_the_hand_rounding_matches_numpys_float16 as aa; aa(); \
from tinyserve.precision import test_each_format_reports_the_range_its_bits_allow as ab; ab(); \
from tinyserve.precision import test_rounding_is_unbiased as ac; ac(); \
from tinyserve.precision import test_a_value_that_fits_is_left_alone as ad; ad(); \
from tinyserve.quantize import test_the_codes_really_fit_in_their_bits as ae; ae(); \
from tinyserve.quantize import test_a_finer_scale_is_never_worse as af; af(); \
from tinyserve.quantize import test_grouping_puts_the_right_values_together as ag; ag(); \
from tinyserve.quantize import test_zero_survives_a_symmetric_scheme as ah; ah(); \
from tinyserve.quantize import test_asymmetric_wins_on_one_sided_data as ai; ai(); \
from tinyserve.speculative import test_the_kept_tokens_have_the_expensive_models_distribution as aj; aj(); \
from tinyserve.speculative import test_a_useless_draft_still_gives_the_right_answer as ak; ak(); \
from tinyserve.speculative import test_a_perfect_draft_is_always_accepted as al; al(); \
from tinyserve.speculative import test_the_expected_tokens_formula_matches_the_simulation as am; am(); \
from tinyserve.speculative import test_more_guesses_stop_paying as an; an(); \
from tinyserve.grammar import test_a_constrained_walk_is_always_valid_json as ao; ao(); \
from tinyserve.grammar import test_without_the_mask_it_is_not_valid as ap; ap(); \
from tinyserve.grammar import test_the_mask_never_allows_nothing as aq; aq(); \
from tinyserve.grammar import test_the_states_are_few_enough_to_precompute as ar; ar(); \
from bench.harness import test_an_unsettled_statistic_is_caught as as_; as_(); \
from tinyserve.cache import test_a_stored_answer_comes_back as at; at(); \
from tinyserve.cache import test_an_omitted_field_serves_the_wrong_tenants_answer as au; au(); \
from tinyserve.cache import test_an_entry_past_its_lifetime_is_a_miss as av; av(); \
from tinyserve.cache import test_eviction_takes_the_least_recently_used as aw; aw(); \
from tinyserve.cache import test_similarity_is_a_similarity as ax; ax(); \
from tinyserve.cache import test_one_decisive_word_scores_higher_than_a_paraphrase as ay; ay(); \
from tinyserve.cache import test_stop_word_folding_merges_opposite_questions as az; az(); \
from tinyserve.cache import test_a_bigger_cache_is_a_likelier_accident as ba; ba(); \
from tinyserve.questions import test_every_topic_has_a_partner_or_is_alone_on_purpose as bb; bb(); \
from tinyserve.questions import test_answers_are_unique_per_topic as bc; bc(); \
from tinyserve.questions import test_each_topic_is_asked_more_than_one_way as bd; bd(); \
from tinyserve.drafting import test_a_medusa_head_is_mostly_its_output_projection as be; be(); \
from tinyserve.drafting import test_a_bigger_head_slows_every_step as bf; bf(); \
from tinyserve.drafting import test_a_chain_is_the_tree_chapter_29_measured as bg; bg(); \
from tinyserve.drafting import test_a_wider_tree_costs_what_it_carries as bh; bh(); \
from tinyserve.drafting import test_breadth_beats_depth_when_the_draft_is_poor as bi; bi(); \
from tinyserve.drafting import test_the_index_proposes_what_followed_before as bj; bj(); \
from tinyserve.drafting import test_extending_the_index_matches_rebuilding_it as bk; bk(); \
from tinyserve.drafting import test_a_drafter_with_nothing_to_copy_proposes_nothing as bl; bl(); \
from tinyserve.evaluate import test_the_exact_test_agrees_with_the_obvious_cases as bm; bm(); \
from tinyserve.evaluate import test_the_paired_test_rejects_at_the_rate_it_claims as bn; bn(); \
from tinyserve.evaluate import test_pairing_finds_what_the_naive_test_cannot as bo; bo(); \
from tinyserve.evaluate import test_the_smallest_detectable_difference_shrinks_with_the_benchmark as bp; bp(); \
from tinyserve.evaluate import test_forty_eight_items_can_only_see_a_large_difference as bq; bq(); \
from tinyserve.evaluate import test_a_continuous_metric_gains_power_with_the_sample as br; br(); \
from tinyserve.evaluate import test_a_fraction_too_small_to_see_is_not_small_per_reply as bs; bs(); \
from tinyserve.evaluate import test_the_largest_difference_is_the_largest_feasible_one as bt; bt(); \
from tinyserve.evaluate import test_impossible_tables_are_refused as bu; bu(); \
from tinyserve.moe import test_the_headline_pair_is_recovered as bv; bv(); \
from tinyserve.moe import test_a_batch_touches_more_experts_than_a_token as bw; bw(); \
from tinyserve.moe import test_the_step_grows_and_the_token_gets_cheaper as bx; bx(); \
from tinyserve.moe import test_spreading_the_experts_divides_only_the_experts as by; by(); \
from tinyserve.moe import test_expert_parallel_traffic_scales_with_the_batch as bz; bz(); \
print('tests: palette ramp, reference model is 8B, listings extract, prefix sharing, batched decode, scheduling, chunked prefill, fleets, attention kernels, the accelerator guard, number formats, quantization, speculation, grammars, steady state, caching above the model, drafting, evaluation, mixtures of experts')"

# One environment per machine the book measures on. The venv is never
# committed; requirements.txt says what to install and the lock records
# exactly what was installed when the numbers in results/ were produced.
venv:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet -r requirements.txt
	@echo "created .venv - run make with PATH=\".venv/bin:$$PATH\", or activate it"

# `make lock` writes the lock for whichever machine you are on. Chapters
# 1-20 were measured on the Linux instance in environment.lock; the
# Apple Silicon numbers from Part IV onward come from environment-mps.lock.
lock:
	@python3 -c "import sys,platform,pathlib,subprocess; \
mps = platform.system() == 'Darwin' and platform.machine() == 'arm64'; \
name = 'environment-mps.lock' if mps else 'environment.lock'; \
froze = subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True).stdout; \
pathlib.Path(name).write_text('# python %s on %s\n%s' % (sys.version.split()[0], platform.platform(), froze)); \
print('wrote ' + name)"

clean:
	rm -rf figures tables results/*.json __pycache__ */__pycache__
