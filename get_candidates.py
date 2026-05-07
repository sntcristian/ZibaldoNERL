import argparse
import os
import csv
import json
from tqdm import tqdm
from src.retriever import EntityDisambiguator, load_disambiguator


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def load_dataset(paragraphs_path, annotations_path):


    with open(paragraphs_path, "r", encoding="utf-8") as doc_f:
        paragraphs = list(csv.DictReader(doc_f))

    with open(annotations_path, "r", encoding="utf-8") as anno_f:
        annotations = list(csv.DictReader(anno_f))

    all_annotations = []
    all_texts = []
    all_offsets = []
    all_lengths = []

    for doc in paragraphs:
        text = doc["text"]
        doc_id = doc["doc_id"]
        doc_anno = [row for row in annotations if row["doc_id"] == doc_id]

        for anno in doc_anno:
            start_pos = int(anno["start_pos"])
            end_pos = int(anno["end_pos"])

            words = text.split()

            word_positions = []
            current_pos = 0
            for word in words:
                word_start = text.find(word, current_pos)
                word_end = word_start + len(word)
                word_positions.append((word_start, word_end))
                current_pos = word_end

            entity_word_idx = None
            for i, (w_start, w_end) in enumerate(word_positions):
                if w_start <= start_pos < w_end:
                    entity_word_idx = i
                    break

            if entity_word_idx is None:
                continue

            chunk_start_word = max(0, entity_word_idx - 50)
            chunk_end_word = min(len(words), entity_word_idx + 50)

            chunk_words = words[chunk_start_word:chunk_end_word]
            chunk_text = " ".join(chunk_words)

            chunk_start_char = word_positions[chunk_start_word][0] if chunk_start_word < len(word_positions) else 0

            new_start_pos = start_pos - chunk_start_char
            new_length = end_pos - start_pos

            if new_start_pos >= 0 and new_start_pos + new_length <= len(chunk_text):
                all_annotations.append(anno)
                all_texts.append([chunk_text])
                all_offsets.append([new_start_pos])
                all_lengths.append([new_length])

    return all_annotations, all_texts, all_offsets, all_lengths


def main():
    parser = argparse.ArgumentParser(description="Entity disambiguation with candidate generation")
    parser.add_argument("--documents", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--annotations", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--lang", type=str, default="it", help="Language code (e.g., 'it', 'en')")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top candidates to retrieve")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of documents in a batch")
    parser.add_argument('--use_hf_model', type=str_to_bool, default=1)
    parser.add_argument('--models_path', type=str, default="./models")
    parser.add_argument('--device', type=str, default="cuda:0")

    args = parser.parse_args()
    
    if args.use_hf_model:
        disambiguator = EntityDisambiguator(hf_model_name="sntcristian/WikiBELA", device=args.device)
    else:
        disambiguator = load_disambiguator(models_path=args.models_path, device=args.device)

    all_annotations, all_texts, all_offsets, all_lengths = load_dataset(args.documents, args.annotations)

    batch_size = args.batch_size
    all_results = []
    for i in tqdm(range(0, len(all_texts), batch_size)):
        batch_texts = [text[0] for text in all_texts[i:i + batch_size]]
        batch_offsets = [offset for offset in all_offsets[i:i + batch_size]]
        batch_lengths = [length for length in all_lengths[i:i + batch_size]]
        batch_annotations = all_annotations[i:i + batch_size]
        batch_predictions = disambiguator.get_candidates_batch(
            batch_texts, batch_offsets, batch_lengths, k=args.top_k, lang=args.lang
        )
        for anno, pred in zip(batch_annotations, batch_predictions):
            all_results.append(
                {
                    "doc_id":anno["doc_id"],
                    "start_pos":anno["start_pos"],
                    "end_pos":anno["end_pos"],
                    "surface":anno["surface"],
                    "type":anno["type"],
                    "candidates":pred
                }
            )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, f"candidates_top{args.top_k}_{args.lang}.json"), "w", encoding="utf-8") as out_f:
        json.dump(all_results, out_f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()