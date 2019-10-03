geniedir := /opt/genie-toolkit
genie := node --experimental-worker --max_old_space_size=30000 $(geniedir)/tool/genie.js

owner ?= gcampax
experiment ?= thingtalk
thingpedia ?= $(experiment)/thingpedia.tt
primitives ?= dataset.tt
paraphrase ?= $(experiment)/paraphrase.tsv
contextual_paraphrase ?= $(tmpdir)/contextual-paraphrasing1.tsv
eval_model ?=

modeldir = models/$(eval_model)
tmpdir = tmp

validation_sets = $(experiment)/validation-contextualized.tsv $(experiment)/validation-turnbyturn.tsv $(experiment)/validation.dialog.txt syntethic-contextual-eval.tsv

thingtalk_synthetic_flags ?= policies aggregation timer projection undefined_filter
spotify_synthetic_flags ?= aggregation timer projection undefined_filter

synthetic_flags = $(foreach v,$($(experiment)_synthetic_flags),--set-flag $(v))
evalflags ?=

all: evaluate

.PHONY: all datadir evaluate

$(tmpdir)/contexts.txt : $(paraphrase) $(tmpdir)/synthetic.tsv $(thingpedia)
	$(genie) extract-contexts -l en-US -o $(tmpdir)/contexts.txt.tmp --thingpedia $(thingpedia) $(paraphrase) $(tmpdir)/synthetic.tsv
	shuf $(tmpdir)/contexts.txt.tmp > $(tmpdir)/contexts.txt

$(tmpdir)/contextualized.tsv : $(tmpdir)/contexts.txt $(paraphrase) $(tmpdir)/synthetic.tsv
	$(genie) contextualize -l en-US -o $(tmpdir)/contextualized.tsv.tmp --expansion-factor 5 -c $(tmpdir)/contexts.txt $(paraphrase) $(tmpdir)/synthetic.tsv
	mv $(tmpdir)/contextualized.tsv.tmp $(tmpdir)/contextualized.tsv

$(tmpdir)/synthetic-d%.tsv : $(thingpedia) entities.json $(primitives) $(geniedir)/languages/en/thingtalk.genie
	$(genie) generate -l en-US -o $@ \
	  --thingpedia $(thingpedia) --entities entities.json --dataset $(primitives) --template $(geniedir)/languages/en/thingtalk.genie \
	  --maxdepth $$(echo $* | cut -f1 -d'-') --no-debug $(synthetic_flags) --set-flag bookkeeping --set-flag no_contextual_bookkeeping --random-seed $@

$(tmpdir)/synthetic-resampled-d5.tsv: $(foreach v,1 2 3 4 5,$(tmpdir)/synthetic-d5-$(v).tsv)
	$(genie) resample -o $@ --fraction 0.2 $^

$(tmpdir)/synthetic.tsv: $(tmpdir)/synthetic-d3.tsv $(tmpdir)/synthetic-resampled-d5.tsv
	cat $^ > $@

$(tmpdir)/synthetic-contextual.tsv : $(tmpdir)/contexts.txt $(thingpedia) entities.json dataset.tt $(geniedir)/languages/en/contextual.genie
	time node --experimental-worker --max_old_space_size=3000 $(geniedir)/tool/genie.js generate-contextual -l en-US -o $(tmpdir)/synthetic-contextual.tsv.tmp \
	  --thingpedia $(thingpedia) --entities entities.json --dataset dataset.tt --template $(geniedir)/languages/en/contextual.genie \
	  $(tmpdir)/contexts.txt --maxdepth 4 --no-debug --parallelize 8 $(synthetic_flags)
	# sort the output file to generate a deterministic output
	sort $(tmpdir)/synthetic-contextual.tsv.tmp -o $(tmpdir)/synthetic-contextual.tsv

$(tmpdir)/easy-hard-functions.tsv: $(tmpdir)/synthetic.tsv
	cut -f3 $(tmpdir)/synthetic.tsv | tr ' ' '\n' | grep -E '^@' | grep -v -E '^@(com|org|gov|uk|us|rest|thermostat|security-camera|edu\.stanford|me\.omlet|car|ca\.randomfox|edu\.mit|light-bulb)\.' | sort -u | sed 's/@/blacklist\t@/g' | cat $(geniedir)/data/easy-hard-functions.tsv - > $@

$(tmpdir)/synthetic-contextual-for-turking.tsv : $(tmpdir)/contexts.txt $(thingpedia) $(primitives) $(geniedir)/languages/en/contextual.genie
	time $(genie) generate-contextual -l en-US -o $(tmpdir)/synthetic-contextual-for-turking.tsv.tmp --thingpedia $(thingpedia) --dataset $(primitives) --template $(geniedir)/languages/en/contextual.genie $(tmpdir)/contexts.txt --maxdepth 4 --no-debug --set-flag turking
	mv $(tmpdir)/synthetic-contextual-for-turking.tsv.tmp $(tmpdir)/synthetic-contextual-for-turking.tsv

$(tmpdir)/synthetic-contextual-sampled.tsv : $(tmpdir)/synthetic-contextual-for-turking.tsv $(tmpdir)/easy-hard-functions.tsv $(tmpdir)/synthetic.tsv $(thingpedia)
	$(genie) sample --thingpedia $(thingpedia) -o $@.tmp --contextual --context-source $(tmpdir)/synthetic.tsv --constants $(geniedir)/data/en-US/constants.tsv --sampling-control easy-hard-functions.tsv --sampling-strategy bySentence $<
	mv $@.tmp $@

$(tmpdir)/mturk-contextual-paraphrasing.csv: $(tmpdir)/synthetic-contextual-sampled.tsv
	cat <(head -n1 $< ) <(tail -n+2 $< | shuf) | $(genie) mturk-make-paraphrase-hits -o $@

$(tmpdir)/contextual-paraphrasing%.tsv: mturk-batches/batch%.csv $(thingpedia)
	$(genie) mturk-validate -o $@ --thingpedia $(thingpedia) --paraphrasing-input $< --validation-threshold 0 --debug --contextual

$(tmpdir)/synthetic-contextual-train.flag: $(tmpdir)/synthetic-contextual.tsv
	touch $@
	$(genie) split-train-eval --train $(tmpdir)/synthetic-contextual-train.tsv --eval $(tmpdir)/synthetic-contextual-eval.tsv --eval-probability 0.05 --split-strategy sentence --contextual --eval-on-synthetic $^

$(tmpdir)/synthetic-contextual-eval.tsv : $(tmpdir)/synthetic-contextual-train.flag

$(tmpdir)/synthetic-contextual-train.tsv : $(tmpdir)/synthetic-contextual-train.flag

$(tmpdir)/everything.tsv : $(tmpdir)/contextualized.tsv $(tmpdir)/synthetic-contextual-train.tsv $(contextual_paraphrase) $(thingpedia) /opt/parameter-datasets/parameter-datasets.tsv
	$(genie) augment -o $@.tmp -l en-US --thingpedia $(thingpedia) --parameter-datasets /opt/parameter-datasets/parameter-datasets.tsv --contextual $(contextual_paraphrase) $(tmpdir)/contextualized.tsv $(tmpdir)/synthetic-contextual-train.tsv \
		--synthetic-expand-factor 1 --quoted-fraction 0.0 --no-replace-locations --no-debug
	mv $@.tmp $@

%-contextualized.tsv : %.tsv
	$(genie) contextualize -l en-US -o $@ --null-only --context contexts.txt $<

%-turnbyturn.tsv : %.dialog.txt $(thingpedia)
	$(genie) dialog-to-contextual -o $@ --thingpedia $(thingpedia) $<

datadir/eval.tsv : $(experiment)/validation-turnbyturn.tsv $(experiment)/validation-contextualized.tsv
	cut -f1-4 $^ >> $@

datadir/train.tsv : $(tmpdir)/everything.tsv
	mkdir -p $(dir $@)
	#$(genie) resample --contextual --fraction 0.2 everything.tsv -o $@
	cp everything.tsv $@

datadir: datadir/train.tsv datadir/eval.tsv
	touch $@

%.dialog.txt.results: %.dialog.txt $(modeldir)/best.pth
	$(genie) evaluate-dialog --url "file://$(modeldir)" --thingpedia $(thingpedia) $< --no-debug --csv $(evalflags) > $@.tmp && mv $@.tmp $@

%.tsv.results: %.tsv $(modeldir)/best.pth
	$(genie) evaluate-server --contextual --url "file://$(modeldir)" --thingpedia $(thingpedia) $< --no-debug --csv $(evalflags) > $@.tmp && mv $@.tmp $@

evaluate: $(foreach v,$(validation_sets),$(v).results)
	cat $^
