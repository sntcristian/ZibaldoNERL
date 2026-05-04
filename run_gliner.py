import os
import argparse
from gliner import GLiNER
import csv
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run GliNER model on DigitalZibaldone")
    parser.add_argument("--documents", type=str, required=True, help="Path to CSV containing documents.")
    parser.add_argument("--output_dir", type=str, default="./results", help="Path to output directory.")
    parser.add_argument("--threshold", type=float, default=0.9, help="Threshold for GliNER model.")
    args = parser.parse_args()
    params = vars(args)

    labels = ["persona", "luogo", "opera"]


    with open(params["documents"], "r", encoding="utf-8") as f:
        paragraphs = csv.DictReader(f)
        paragraphs = list(paragraphs)
    f.close()

    model = GLiNER.from_pretrained("sntcristian/GliNER_ENEIDE")
    
    result = list()
    
    for par in tqdm(paragraphs):
        doc_id, text = par["doc_id"], par["text"]
        entities = model.predict_entities(text, labels, threshold=args["threshold"])
        for e in entities:
            item = {"doc_id":doc_id, "surface": e["text"], "start_pos":e["start"], "end_pos":e["end"], "type": e["label"]}
            result.append(item)

    if not os.path.exists(params["output_dir"]):
        os.makedirs(params["output_dir"])

    with open(os.path.join(params["output_dir"], "output_ner.csv"), "w", encoding="utf-8") as f:
        dict_writer = csv.DictWriter(f, result[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(result)

if __name__ == "__main__":
    main()