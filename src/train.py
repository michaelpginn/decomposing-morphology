from typing import Optional, List
import fire
import datasets
from transformers import MT5Tokenizer, DataCollatorForSeq2Seq
from functools import reduce
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm
import wandb
import random
from model import StructuredGlossTransformer
import eval
import features

MAX_INPUT_LENGTH = 64
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 64
MAX_EPOCHS = 125

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
print(device)


def eval_epoch(model: StructuredGlossTransformer, dataloader: DataLoader, labels: List[str], gloss_to_omit: Optional[str]):
    model.eval()
    losses = 0

    preds = []
    gold = []

    for batch in tqdm(dataloader, desc="Eval batch", colour="green"):
        batch_features = features.map_labels_to_features(batch['labels'], model.feature_map)
        (loss, feature_logits) = model.forward(input_ids=batch['input_ids'].to(device),
                                               attention_mask=batch['attention_mask'].to(device),
                                               decoder_input_ids=batch['labels'].to(device),
                                               features=batch_features.to(device))
        losses += loss.item()
        decoded_tokens = model.greedy_decode(feature_logits=feature_logits).cpu()
        preds += [[labels[index] for index in seq] for seq in decoded_tokens]
        gold += [[labels[index] for index in seq if index >= 0] for seq in batch['labels']]

        preds = [seq[:len(gold[row_index])] for row_index, seq in enumerate(preds)]

    if gloss_to_omit is not None:
        # Find and display rows with the gloss in question
        for p, g in zip(preds, gold):
            if gloss_to_omit in g:
                print(f"Pred: {p}\nGold: {g}\n")
    else:
        # Print the first few
        for p, g in zip(preds[:10], gold[:10]):
            print(f"Pred: {p}\nGold: {g}\n")

    metrics = {
        **eval.eval_accuracy(preds, gold),
        **eval.bleu(preds, gold)
    }

    return losses / len(dataloader), metrics


def test(model: StructuredGlossTransformer, dataloader: DataLoader, labels):
    model.eval()
    preds = []
    gold = []

    for batch in tqdm(dataloader, desc="Test batch", colour="green"):
        tokens = model.generate(input_ids=batch['input_ids'].to(device),
                                attention_mask=batch['attention_mask'].to(device)).cpu()
        preds += [[labels[index] for index in seq] for seq in tokens]
        gold += [[labels[index] for index in seq if index >= 0] for seq in batch['labels']]

        # Trim preds
        preds = [seq[:len(gold[row_index])] for row_index, seq in enumerate(preds)]

    metrics = {
        **eval.eval_accuracy(preds, gold),
        **eval.bleu(preds, gold)
    }
    return preds, metrics


def main(mode: str = 'train',
         language: str = 'usp',
         features_path: Optional[str] = None,
         model_path: Optional[str] = None,
         gloss_to_omit: Optional[str] = None,
         seed: int = 0):
    """Runs the training code

    Args:
        mode (str, optional): 'train' | 'eval' | 'test'
        language (str, optional): 'usp' | 'ddo'
        features_path (Optional[str], optional): Path to features CSV file. If not provided, use baseline model.
        model_path (Optional[str], optional): Path to trained model `.pth` file, for inference.
        gloss_to_omit (Optional[str], optional): If provided, omits the specified gloss from the training set.
        seed (int, optional): Random seed.

    Returns:
        _type_: _description_
    """
    random.seed(seed)
    experiment = "baseline" if features_path is None else "structured",
    wandb.init(
        project="decomposing-morphology",
        config={
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "experiment": experiment,
            "language": language,
            "feature_map": features_path,
            "held_out_gloss": gloss_to_omit
        }
    )

    if language == 'usp':
        dataset = datasets.load_dataset("lecslab/usp-igt")
    else:
        dataset = datasets.load_dataset("lecslab/ddo-igt", download_mode="force_redownload")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", legacy=False)

    if gloss_to_omit is not None:
        # Find all of the rows with that tag in the train set and move to test
        filtered_rows = dataset['train'].filter(
            lambda r: gloss_to_omit in r['pos_glosses'].replace('-', ' ').split())
        dataset['train'] = dataset['train'].filter(
            lambda r: gloss_to_omit not in r['pos_glosses'].replace('-', ' ').split())
        dataset['test'] = datasets.concatenate_datasets([dataset['test'], filtered_rows])
        dataset['eval'] = datasets.concatenate_datasets([dataset['eval'], filtered_rows])

    # Collect the unique set of gloss labels
    all_glosses = sorted(set([gloss for glosses in dataset['train']['pos_glosses'] +
                              dataset['eval']['pos_glosses'] +
                              dataset['test']['pos_glosses'] for gloss in glosses.replace("-", " ").split()]))
    all_glosses = ["<sep>", "<pad>"] + all_glosses
    SEP_TOKEN_ID = all_glosses.index("<sep>")
    PAD_TOKEN_ID = all_glosses.index("<pad>")
    print(f"{len(all_glosses)} unique glosses")

    # Create a feature map based on a table of glosses x features
    if features_path:
        feature_map, feature_lengths = features.create_feature_map(features_path, all_glosses)
    else:
        feature_map = [[i] for i in range(len(all_glosses))]
        feature_lengths = [len(all_glosses)]

    def encode_gloss_labels(label_string: str):
        """Encodes glosses as an id sequence. Each morpheme gloss is assigned a unique id."""
        word_glosses = label_string.split()
        glosses = [word_gloss.split("-") for word_gloss in word_glosses]
        glosses = [[all_glosses.index(gloss) for gloss in word if gloss != ''] for word in glosses]
        glosses = reduce(lambda a, b: a + [SEP_TOKEN_ID] + b, glosses)
        return glosses + [PAD_TOKEN_ID]

    def tokenize(batch):
        inputs = tokenizer(batch['transcription'], truncation=True, padding=False, max_length=MAX_INPUT_LENGTH)
        inputs['labels'] = [encode_gloss_labels(label) for label in batch['pos_glosses']]
        return inputs

    dataset = dataset.map(tokenize, batched=True).select_columns(['input_ids', 'attention_mask', 'labels'])
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    dataloader = DataLoader(dataset['train'], batch_size=BATCH_SIZE, collate_fn=collator)
    eval_dataloader = DataLoader(dataset['eval'], batch_size=BATCH_SIZE, collate_fn=collator)
    test_dataloader = DataLoader(dataset['test'], batch_size=BATCH_SIZE, collate_fn=collator)

    # Create the model. For all experiments, the first "feature" is just a unique id for each gloss
    # A trivial "feature map" baseline, with one layer of features (where each option is a unique gloss)
    model = StructuredGlossTransformer(num_glosses=len(all_glosses),
                                       feature_lengths=feature_lengths,
                                       feature_map=feature_map,
                                       decoder_pad_token_id=PAD_TOKEN_ID,
                                       decoder_max_length=MAX_OUTPUT_LENGTH).to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    if mode == 'train':
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # Training loop
        for epoch in tqdm(range(MAX_EPOCHS), desc="Epoch", colour="blue"):
            start_time = timer()

            model.train()
            losses = 0

            for batch in tqdm(dataloader, desc="Batch", colour="green"):
                batch_features = features.map_labels_to_features(batch['labels'], feature_map)
                (loss, feature_logits) = model.forward(input_ids=batch['input_ids'].to(device),
                                                       attention_mask=batch['attention_mask'].to(device),
                                                       decoder_input_ids=batch['labels'].to(device),
                                                       features=batch_features.to(device))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            end_time = timer()
            train_loss = losses / len(dataloader)

            # Eval
            eval_loss, metrics = eval_epoch(model, eval_dataloader, all_glosses, gloss_to_omit)

            print(
                (f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Eval loss: {eval_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

            wandb.log({"epoch": epoch,
                       "train": {
                           "loss": train_loss
                       },
                       "eval": {
                           "loss": eval_loss,
                           **metrics
                       }})

    if mode == 'eval':
        eval_loss, metrics = eval_epoch(model, eval_dataloader, all_glosses)
        print(metrics)

    if mode == 'train' or mode == 'test':
        # Run final eval on test
        test_preds, test_metrics = test(model, test_dataloader, all_glosses)
        wandb.log({
            "test": test_metrics
        })
        with open(f"preds-{experiment}-{seed}-{gloss_to_omit if gloss_to_omit is not None else 'all'}.out", 'w') as f:
            for seq in test_preds:
                # Join the strings in the inner list with a space and write to the file
                f.write(' '.join(seq) + '\n')

    wandb.finish()
    torch.save(model.state_dict(), "./model.pth")


if __name__ == "__main__":
    fire.Fire(main)
