import time
from tika import parser

def extract_text(pdf_path):
    raw = parser.from_file(pdf_path)
    content = raw["content"]

    txt_output_path = "./output/output.txt"
    with open(txt_output_path, "w") as f:
        f.write(content)

def main():
    pdf_path = "./output/cropped_output.pdf"
    start_time = time.time()
    extract_text(pdf_path)
    end_time = time.time()
    print(f"Tempo impiegato: {end_time - start_time} secondi")
    

if __name__ == "__main__":
    main()