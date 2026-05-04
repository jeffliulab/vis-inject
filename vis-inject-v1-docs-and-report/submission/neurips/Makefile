# report/submission/neurips/Makefile --- build the NeurIPS-format
# version of the VisInject paper.
#
# Source content is mirrored from report/xarchive-report/paper/ (do not
# modify that tree). This Makefile only builds main.pdf in this folder.
#
# Targets:
#   make / make pdf   Build main.pdf via latexmk (default).
#   make watch        Continuous rebuild (latexmk -pvc).
#   make clean        Remove aux files (keeps main.pdf).
#   make distclean    Also remove main.pdf.

LATEXMK ?= latexmk
MAIN    := main

.PHONY: all pdf watch clean distclean

all: pdf

pdf:
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error $(MAIN).tex

watch:
	$(LATEXMK) -pdf -pvc -bibtex -interaction=nonstopmode $(MAIN).tex

clean:
	$(LATEXMK) -c
	rm -f *.aux *.log *.bbl *.blg *.fls *.fdb_latexmk *.out *.toc *.lof *.lot *.synctex.gz *.nav *.snm *.vrb *.run.xml

distclean: clean
	rm -f $(MAIN).pdf
