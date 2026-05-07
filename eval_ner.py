import csv
import argparse
import os


def compute_match(annotation1, annotation2, match_type):
    start_pos1 = int(annotation1["start_pos"])
    end_pos1 = int(annotation1["end_pos"])
    ent_type1 = annotation1["type"]
    start_pos2 = int(annotation2["start_pos"])
    end_pos2 = int(annotation2["end_pos"])
    ent_type2 = annotation2["type"]
    if match_type=="exact":
        if start_pos1==start_pos2 and end_pos1==end_pos2 and ent_type1 == ent_type2:
            return True
        else:
            return False
    elif match_type=="relaxed":
        char_intersection = len(set(range(start_pos1, end_pos1)).intersection(set(range(start_pos2, end_pos2))))
        if char_intersection > 0 and ent_type1 == ent_type2:
            return True
        else:
            return False
    else:
        print("Wrong match type: use exact or relaxed as values")
        return None


def eval_ner(data, model_result, match_type):
    tp = []  # true positive
    fp = []  # false positive
    fn = []  # false negative
    matches = []  # matched annotations
    for entity1 in data:
        id1 = entity1["doc_id"]
        for entity2 in model_result:
            id2 = entity2["doc_id"]
            if id1==id2:
                match_value = compute_match(entity1, entity2, match_type)
                if match_value==True:
                    matches.append(entity1)
                    tp.append(entity2)
                    break

    for entity1 in data:
        if entity1 not in matches:
            fn.append(entity1)

    for entity2 in model_result:
        if entity2 not in tp:
            fp.append(entity2)

    precision = len(matches) / (len(matches) + len(fp))
    recall = len(matches) / (len(matches) + len(fn))
    f1 = (2 * precision * recall) / (precision + recall)
    return [len(tp), len(fp), len(fn), precision, recall, f1]


def main():
    parser = argparse.ArgumentParser(description="NER Evaluation script")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV containing gold annotations.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to folder containing NER output")
    args = parser.parse_args()
    params = vars(args)


    with open(params["data"], "r", encoding="utf-8") as f2:
        data = csv.DictReader(f2)
        data = list(data)
    f2.close()

    with open(os.path.join(params["output_dir"],"output_ner"), "r", encoding="utf-8") as f3:
        model_result = csv.DictReader(f3)
        model_result = list(model_result)
    f3.close()


    data_per = [row for row in data if row["type"] == "PER"]
    model_result_per = [row for row in model_result if row["type"] == "PER" ]

    data_loc = [row for row in data if row["type"] == "LOC"]
    model_result_loc = [row for row in model_result if row["type"] == "LOC" ]

    data_work = [row for row in data if row["type"] == "WORK"]
    model_result_work = [row for row in model_result if row["type"] == "WORK" ]

    results_exact = eval_ner(data, model_result, "exact")
    results_relaxed = eval_ner(data, model_result, "relaxed")

    results_per_exact = eval_ner(data_per, model_result_per, "exact")
    results_per_relaxed = eval_ner(data_per, model_result_per, "relaxed")

    results_work_exact = eval_ner(data_work, model_result_work, "exact")
    results_org_relaxed = eval_ner(data_work, model_result_work, "relaxed")

    results_loc_exact = eval_ner(data_loc, model_result_loc, "exact")
    results_loc_relaxed = eval_ner(data_loc, model_result_loc, "relaxed")

    with open(params["output_dir"] + "results_ner.txt", "w") as output:
        output.write("Results with exact match for all classes:\n\n")
        output.write("True Positives: " + str(results_exact[0]) + "\n")
        output.write("False Positives: " + str(results_exact[1]) + "\n")
        output.write("False Negatives: " + str(results_exact[2]) + "\n")
        output.write("Precision: " + str(results_exact[3]) + "\n")
        output.write("Recall: " + str(results_exact[4]) + "\n")
        output.write("F1: " + str(results_exact[5]) + "\n\n")

        output.write("Results with relaxed match for all classes:\n\n")
        output.write("True Positives: " + str(results_relaxed[0]) + "\n")
        output.write("False Positives: " + str(results_relaxed[1]) + "\n")
        output.write("False Negatives: " + str(results_relaxed[2]) + "\n")
        output.write("Precision: " + str(results_relaxed[3]) + "\n")
        output.write("Recall: " + str(results_relaxed[4]) + "\n")
        output.write("F1: " + str(results_relaxed[5]) + "\n\n")

        output.write("Results with exact match for class Person:\n\n")
        output.write("True Positives: " + str(results_per_exact[0]) + "\n")
        output.write("False Positives: " + str(results_per_exact[1]) + "\n")
        output.write("False Negatives: " + str(results_per_exact[2]) + "\n")
        output.write("Precision: " + str(results_per_exact[3]) + "\n")
        output.write("Recall: " + str(results_per_exact[4]) + "\n")
        output.write("F1: " + str(results_per_exact[5]) + "\n\n")

        output.write("Results with relaxed match for class Person:\n\n")
        output.write("True Positives: " + str(results_per_relaxed[0]) + "\n")
        output.write("False Positives: " + str(results_per_relaxed[1]) + "\n")
        output.write("False Negatives: " + str(results_per_relaxed[2]) + "\n")
        output.write("Precision: " + str(results_per_relaxed[3]) + "\n")
        output.write("Recall: " + str(results_per_relaxed[4]) + "\n")
        output.write("F1: " + str(results_per_relaxed[5]) + "\n\n")

        output.write("Results with exact match for class Work:\n\n")
        output.write("True Positives: " + str(results_work_exact[0]) + "\n")
        output.write("False Positives: " + str(results_work_exact[1]) + "\n")
        output.write("False Negatives: " + str(results_work_exact[2]) + "\n")
        output.write("Precision: " + str(results_work_exact[3]) + "\n")
        output.write("Recall: " + str(results_work_exact[4]) + "\n")
        output.write("F1: " + str(results_work_exact[5]) + "\n\n")

        output.write("Results with relaxed match for class Work:\n\n")
        output.write("True Positives: " + str(results_org_relaxed[0]) + "\n")
        output.write("False Positives: " + str(results_org_relaxed[1]) + "\n")
        output.write("False Negatives: " + str(results_org_relaxed[2]) + "\n")
        output.write("Precision: " + str(results_org_relaxed[3]) + "\n")
        output.write("Recall: " + str(results_org_relaxed[4]) + "\n")
        output.write("F1: " + str(results_org_relaxed[5]) + "\n\n")

        output.write("Results with exact match for class Location:\n\n")
        output.write("True Positives: " + str(results_loc_exact[0]) + "\n")
        output.write("False Positives: " + str(results_loc_exact[1]) + "\n")
        output.write("False Negatives: " + str(results_loc_exact[2]) + "\n")
        output.write("Precision: " + str(results_loc_exact[3]) + "\n")
        output.write("Recall: " + str(results_loc_exact[4]) + "\n")
        output.write("F1: " + str(results_loc_exact[5]) + "\n\n")

        output.write("Results with relaxed match for class Location:\n\n")
        output.write("True Positives: " + str(results_loc_relaxed[0]) + "\n")
        output.write("False Positives: " + str(results_loc_relaxed[1]) + "\n")
        output.write("False Negatives: " + str(results_loc_relaxed[2]) + "\n")
        output.write("Precision: " + str(results_loc_relaxed[3]) + "\n")
        output.write("Recall: " + str(results_loc_relaxed[4]) + "\n")
        output.write("F1: " + str(results_loc_relaxed[5]) + "\n\n")

if __name__ == "__main__":
    main()