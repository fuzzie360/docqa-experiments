
import sys

import argparse
import torch
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdf2image import convert_from_path
from transformers import AutoTokenizer, LayoutLMForQuestionAnswering

parser = argparse.ArgumentParser(
    prog='docqa.py',
    description='Ask questions about a PDF',
)

parser.add_argument('inputFile', required=True)
parser.add_argument('question', required=True)

args = parser.parse_args()

question = args.question
inputFile = args.inputFile

model_checkpoint = "impira/layoutlm-document-qa"
model_revision = "1e3ebac"

pages = convert_from_path(inputFile, 500)
if len(pages) > 1:
    print("Multiple page PDF not supported")
    sys.exit()

image = pages[0]

fp = open(inputFile, 'rb')
rsrcmgr = PDFResourceManager()
laparams = LAParams()
device = PDFPageAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)
pages = PDFPage.get_pages(fp)

def create_bounding_box(bbox_data):
    xs = []
    ys = []
    for x, y in bbox_data:
        xs.append(x)
        ys.append(y)
 
    left = int(min(xs))
    top = int(min(ys))
    right = int(max(xs))
    bottom = int(max(ys))
 
    return [left, top, right, bottom]

def scale_bounding_box(box, width_scale = 1.0, height_scale = 1.0):
    return [
        int(box[0] * width_scale),
        int(box[1] * height_scale),
        int(box[2] * width_scale),
        int(box[3] * height_scale)
    ]

words = []
boxes = []
page = next(pages) # pages is a generator
_, _, width, height = page.mediabox
width_scale = 1000 / width
height_scale = 1000 / height
interpreter.process_page(page)

layout = device.get_result()
for lobj in layout:
    if isinstance(lobj, LTTextBox):
        boxes.append(
            scale_bounding_box(
                lobj.bbox,
                width_scale,
                height_scale
            )
        )
        words.append(lobj.get_text())

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)
model = LayoutLMForQuestionAnswering.from_pretrained(model_checkpoint, revision=model_revision)

encoding = tokenizer(
    question.split(), words, is_split_into_words=True, return_token_type_ids=True, return_tensors="pt"
)
bbox = []
for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
    if s == 1:
        bbox.append(boxes[w])
    elif i == tokenizer.sep_token_id:
        bbox.append([1000] * 4)
    else:
        bbox.append([0] * 4)
encoding["bbox"] = torch.tensor([bbox])

word_ids = encoding.word_ids(0)
outputs = model(**encoding)
loss = outputs.loss
start_scores = outputs.start_logits
end_scores = outputs.end_logits
start, end = word_ids[start_scores.argmax(-1)], word_ids[end_scores.argmax(-1)]
print(" ".join(words[start : end + 1]))
