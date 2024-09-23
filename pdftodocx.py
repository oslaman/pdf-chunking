from pdf2docx import Converter

pdf_file  = r'./sample8.pdf'# source file 
docx_file = r'./sample8.docx'  # destination file

# convert pdf to docx
cv = Converter(pdf_file)
cv.convert(docx_file, start=0, end=None)
cv.close()