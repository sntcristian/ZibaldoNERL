import json
import csv
import transformers
import torch
import re
from tqdm import tqdm
import os
import argparse


def process_candidates(candidates, n_candidates):
    output = []
    candidates_new = [c for c in candidates if len(c["label"])>0]
    for item in candidates_new[:min(n_candidates, len(candidates_new))]:
        output.append({
            "wikipedia_page": item["label"],
            "wikidata_id": item["wb_id"],
            "type": item["type"],
            "descr":item["descr"],
            "date": item["min_date"],
        })
    return output


def main():
    parser = argparse.ArgumentParser(description="LLM Prompting for Entity Disambiguation from Candidate List")
    parser.add_argument("--json_f", type=str, required=True, help="Path to JSON list of candidates")
    parser.add_argument("--documents_path", type=str, required=True, help="Path to documents")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold to be used to filter hard negatives.")
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-Small-24B-Instruct-2501", help="Huggingface repo of LLM")
    parser.add_argument("--hf_token", type=str, default="", help="Huggingface token to access restricted repo.")
    parser.add_argument("--n_candidates", type=int, default=10, help="Number of candidates to put in prompt.")

    args = parser.parse_args()
    with open(args.json_f, "r", encoding="utf-8") as f:
        retriever_results = json.load(f)

    with open(args.documents_path, "r", encoding="utf-8") as f:
        paragraphs = list(csv.DictReader(f))


    pipeline = transformers.pipeline(
        "text-generation",
        model=args.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=args.hf_token
    )

    system_prompt = """Sei un efficace sistema multilingue di estrazione delle informazioni specializzato nella disambiguazione di entità all'interno di testi della letteratura italiana.
    Il tuo compito è analizzare il testo fornito dall'utente e disambiguare il riferimento contrassegnato dai tag <ENT></ENT> selezionando un'entità Wikidata da una lista di candidati fornita solo quando presente, classificandola con un valore NIL in caso contrario.
    Rispondi sempre restituendo una risposta in formato JSON; non generare codice Python."""

    output = []
    paragraphs_dict = {p["doc_id"]: p for p in paragraphs}

    for item in tqdm(retriever_results):
        doc_id = item["doc_id"]
        start_pos = int(item["start_pos"])
        end_pos = int(item["end_pos"])
        if args.threshold > 0 and item["candidates"][0]["score"] >= args.threshold:
            output.append({
                "doc_id":doc_id,
                "start_pos":start_pos,
                "end_pos":end_pos,
                "surface":item["surface"],
                "type":item["type"],
                "identifier":item["candidates"][0]["wb_id"],
                "label":item["candidates"][0]["label"],
                "answer":"",
                "score":0
            })
        else:
            paragraph = paragraphs_dict[doc_id]
            text = paragraph["text"]
            processed_text = text[max(0, start_pos - 500):start_pos] + "<ENT> " + text[start_pos:end_pos] + " <ENT> " +text[end_pos:min(len(text), end_pos + 500)]
            processed_candidates = process_candidates(item["candidates"], args.n_candidates)
            user_prompt = """
            Analizza attentamente il testo estratto da una pagina dello Zibaldone di pensieri di Giacomo Leopardi.
            Disambiguate the entity mentioned between the [ENT] tags by selecting the most appropriate Wikidata entity from the list of candidates.    
            Return the corresponding Wikipedia page title and Wikidata ID of the selected entity in a JSON object formatted as follows:
        
            ```json
            {"wikipedia_page":"", "wikidata_id":""}
            ```
        
            Make sure to select both the Wikidata ID and the Wikipedia page title from the provided list of candidates.
            Pay attention that the list of candidates may not include the entity mentioned. If none of the candidates match with high confidence the entity tagged with [ENT], use the string "NIL" as value of the "wikidata_id" key.
            ---------------------
            Input Text:
            """ + processed_text + """
            ---------------------
            JSON List of Candidates:
            ```json
            """ + str(processed_candidates) + """ 
            ``` ."""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            outputs = pipeline(
                messages,
                max_new_tokens=256,
            )
            response = outputs[0]["generated_text"][-1]["content"]
            match = re.search(r'"wikidata_id"\s*:\s*"(Q\d+)"', response)

            if match:
                wikidata_id = match.group(1)
            else:
                wikidata_id = "NIL"


            selected_entity = [x for x in item["candidates"] if x["wb_id"] == wikidata_id]
            if len(selected_entity) > 0:
                output.append({
                    "doc_id":doc_id,
                    "start_pos":start_pos,
                    "end_pos":end_pos,
                    "surface":item["surface"],
                    "type":item["type"],
                    "identifier":wikidata_id,
                    "label":selected_entity[0]["label"],
                    "answer":re.sub(r'\s+', " ", response),
                    "score":selected_entity[0]["score"]
                })
            else:
                output.append({
                    "doc_id": doc_id,
                    "start_pos": start_pos,
                    "end_pos": end_pos,
                    "surface": item["surface"],
                    "type": item["type"],
                    "identifier": "NIL",
                    "label": item["surface"],
                    "answer": re.sub(r'\s+', " ", response),
                    "score": 0
                })
    
                
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "output_nel.csv"), "w", encoding="utf-8") as out_f:
        dict_writer = csv.DictWriter(out_f, output[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(output)

if __name__ == "__main__":
    main()


