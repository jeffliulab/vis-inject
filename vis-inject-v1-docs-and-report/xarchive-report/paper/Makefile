# report/pdf/Makefile — build the VisInject final report PDF.
#
# Targets:
#   make / make pdf   Build main.pdf via latexmk (default).
#   make figures      Regenerate matplotlib figures in figures/.
#   make watch        Continuous rebuild (latexmk -pvc).
#   make clean        Remove aux files (keeps main.pdf).
#   make distclean    Also remove main.pdf and generated figures.

LATEXMK    ?= latexmk
PYTHON     ?= python3
MAIN       := main
SCRIPTS    := ../scripts

.PHONY: all pdf figures watch clean distclean

all: pdf

pdf:
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error $(MAIN).tex

figures:
	$(PYTHON) $(SCRIPTS)/build_triptych.py
	$(PYTHON) $(SCRIPTS)/build_per_vlm_bar.py

watch:
	$(LATEXMK) -pdf -pvc -bibtex -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEXMK) -c
	rm -f *.aux *.log *.bbl *.blg *.fls *.fdb_latexmk *.out *.toc *.lof *.lot *.synctex.gz *.nav *.snm *.vrb *.run.xml

distclean: clean
	rm -f $(MAIN).pdf
	rm -f figures/triptych.pdf figures/per_vlm.pdf
