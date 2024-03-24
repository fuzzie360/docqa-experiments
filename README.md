
# docqa.py

## Description

Ask questions about a PDF

## Installatioon

Install pre-requsites: 

```
# Windows: https://sourceforge.net/projects/poppler-win32/
# OSX:     brew install poppler
# Linux:   sudo apt install poppler-utils
```

Install python dependencies:

```
pip install -r requirements.txt
```

## Usage

```
> python docqa.py input.pdf "What is the title of this document?"
PAYSLIP ADVICE
```

## Todo

* Use something better than pdfminer and pdf2image
* Context size management