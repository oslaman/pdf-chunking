from tika import parser
import re
from collections import Counter

def extract_text(pdf_path):
    raw = parser.from_file(pdf_path)
    content = raw["content"]

    # txt_output_path = "./output/output.txt"
    # with open(txt_output_path, "w") as f:
    #     f.write(content)

    return content


def rank_most_common_sentences(text, top_n=10):
    sentences = sentences = re.split(r'(?<=[.!?]) +', text)
    sentence_counts = Counter(sentences)
    most_common_sentences = sentence_counts.most_common(top_n)

    return most_common_sentences

print(rank_most_common_sentences(extract_text("./examples/fritzeng.pdf")))