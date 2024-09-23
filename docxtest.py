from docx import Document
from tika import parser

input_file = 'sample.docx'
output_file = 'cleaned_sample.docx'

parser = parser.from_file(input_file)
extracted_text = parser['content']


document = Document()
document.add_paragraph(extracted_text)
document.save(output_file)

document.save("cleaned.doc")