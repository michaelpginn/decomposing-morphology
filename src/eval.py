from typing import List
from torchtext.data.metrics import bleu_score


def eval_accuracy(pred: List[List[str]], gold: List[List[str]]) -> dict:
    """Computes the average and overall accuracy, where predicted labels must be
    in the correct position in the list. Ignores any tokens longer than the gold sequence."""
    total_correct_predictions = 0
    total_tokens = 0
    summed_accuracies = 0

    for entry_pred, entry_gold, i in zip(pred, gold, range(len(gold))):
        entry_correct_predictions = 0

        for token_index in range(len(entry_gold)):
            # For each token, check if it matches
            if (
                token_index < len(entry_pred)
                and entry_pred[token_index] == entry_gold[token_index]
                and entry_pred[token_index] != "[UNK]"
            ):
                entry_correct_predictions += 1

        entry_accuracy = entry_correct_predictions / len(entry_gold)
        summed_accuracies += entry_accuracy

        total_correct_predictions += entry_correct_predictions
        total_tokens += len(entry_gold)

    total_entries = len(gold)
    average_accuracy = summed_accuracies / total_entries
    overall_accuracy = total_correct_predictions / total_tokens
    return {"average_accuracy": average_accuracy, "accuracy": overall_accuracy}


def bleu(pred: List[List[str]], gold: List[List[str]]):
    return {"bleu": bleu_score(pred, [[line] for line in gold])}
