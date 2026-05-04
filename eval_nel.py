import csv
import os


result_path = "./results"
annotations_path = "./data/annotations_test.csv"


def compute_match(annotation1, annotation2, match_type):
    start_pos1 = int(annotation1["start_pos"])
    end_pos1 = int(annotation1["end_pos"])
    wiki_id1 = annotation1["identifier"]
    if not wiki_id1.startswith("Q"):
        wiki_id1 = "NIL"
    start_pos2 = int(annotation2["start_pos"])
    end_pos2 = int(annotation2["end_pos"])
    wiki_id2 = annotation2["identifier"]
    if match_type=="exact":
        if start_pos1==start_pos2 and wiki_id1 == wiki_id2:
            return True
        else:
            return False
    elif match_type=="relaxed":
        char_intersection = len(set(range(start_pos1, end_pos1)).intersection(set(range(start_pos2, end_pos2))))
        if char_intersection > 0 and wiki_id1 == wiki_id2:
            return True
        else:
            return False
    else:
        print("Wrong match type: use exact or relaxed as values")
        return None


def eval_nel(data, model_result, match_type):
    tp = []  
    fp = []  
    fn = []  
    matches = []  
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

    precision = (len(matches) / (len(matches) + len(fp)))*100
    recall = (len(matches) / (len(matches) + len(fn)))*100
    f1 = (2 * precision * recall) / (precision + recall)
    return [len(tp), len(fp), len(fn), precision, recall, f1]



with open(annotations_path, "r", encoding="utf-8") as f2:
    data = csv.DictReader(f2)
    data = list(data)
f2.close()

with open(os.path.join(result_path, "output_nel.csv"), "r", encoding="utf-8") as f3:
    model_result = csv.DictReader(f3)
    model_result = list(model_result)
f3.close()


results_exact = eval_nel(data, model_result, "exact")
results_relaxed = eval_nel(data, model_result, "relaxed")


with open(os.path.join(result_path, "results.txt"), "w") as output:
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

output.close()



