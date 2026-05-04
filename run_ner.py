import csv
from gliner import GLiNER
from tqdm import tqdm

INPUT_CSV  = "../data/letters_text.csv"
OUTPUT_CSV = "../data/letters_ner.csv"
MODEL_PATH = "gliner_all_b4_e4"   
LABELS     = ["persona"]
THRESHOLD  = 0.5

model = GLiNER.from_pretrained(MODEL_PATH, local_files_only=True)
model = model.to("cuda")
model.data_processor.config.max_len = 764

with open(INPUT_CSV, encoding="utf-8") as fin, \
     open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:

    reader = csv.DictReader(fin)
    writer = csv.writer(fout)
    writer.writerow(["par_id", "surface_form", "type", "start_offset", "end_offset"])

    for row in tqdm(reader):
        par_id, text = row["par_id"], row["par_text"]
        entities = model.predict_entities(text, LABELS, threshold=THRESHOLD)
        for e in entities:
            writer.writerow([par_id, e["text"], e["label"], e["start"], e["end"]])

print(f"NER completata, risultati in {OUTPUT_CSV}")