# VisInject — EE141 Final Submission

Three files, one folder.

| File | What it is |
|---|---|
| `VisInject_Final_Report.pdf` | Written report, 18 pages, native LaTeX. |
| `VisInject_Slides.pptx`      | Presentation deck, 25 slides, 16:9 widescreen. |
| `VisInject_Code.zip`         | Snapshot of the project source code at submission time (excludes this `report/submission/` folder, model weights, and HuggingFace cache to keep the archive lean). |

## Project links

- Code: <https://github.com/jeffliulab/vis-inject>
- Dataset: <https://huggingface.co/datasets/jeffliulab/visinject>
- Demo: <https://huggingface.co/spaces/jeffliulab/visinject>

## How to rebuild these artefacts from the code

From the project root:

```bash
# Rebuild the slide deck
python report/scripts/build_slides.py
# → writes report/slides/VisInject_final.pptx

# Rebuild the PDF report
cd report/pdf && make
# → writes report/pdf/main.pdf
```

Build prerequisites:

- TeX Live or MacTeX (`pdflatex`, `latexmk`, `bibtex`)
- Python 3.10+ with `python-pptx`, `matplotlib`, `Pillow` (see `report/scripts/requirements.txt`)

## Author

Pang Liu — `pang.liu@tufts.edu` (or `jeff.pang.liu@gmail.com` after graduation)
