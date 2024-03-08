import fire
import datasets
from transformers import MT5Tokenizer, DataCollatorForSeq2Seq
from functools import reduce
import torch
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm import tqdm
from model import StructuredGlossTransformer
import wandb
import random
import eval

MAX_INPUT_LENGTH = 64
MAX_OUTPUT_LENGTH = 64
BATCH_SIZE = 64
MAX_EPOCHS = 125

device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')
print(device)


def eval_epoch(model: StructuredGlossTransformer, dataloader: DataLoader, labels):
    model.eval()
    losses = 0

    preds = []
    gold = []

    for batch in tqdm(dataloader, desc="Eval batch", colour="green"):
        (loss, feature_logits) = model.forward(input_ids=batch['input_ids'].to(device),
                                               attention_mask=batch['attention_mask'].to(device),
                                               decoder_input_ids=batch['labels'].to(device),
                                               features=batch['labels'].to(device).unsqueeze(-2))
        losses += loss.item()
        decoded_tokens = model.greedy_decode(feature_logits=feature_logits).cpu()
        preds += [[labels[index] for index in seq] for seq in decoded_tokens]
        gold += [[labels[index] for index in seq if index >= 0] for seq in batch['labels']]

        preds = [seq[:len(gold[row_index])] for row_index, seq in enumerate(preds)]

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


def main():
    random.seed(0)
    wandb.init(
        # set the wandb project where this run will be logged
        project="decomposing-morphology",

        # track hyperparameters and run metadata
        config={
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "experiment": "baseline"
        }
    )

    dataset = datasets.load_dataset("lecslab/usp-igt")
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small", legacy=False)

    # Collect the unique set of gloss labels
    all_glosses = sorted(set([gloss for glosses in dataset['train']['pos_glosses'] +
                              dataset['eval']['pos_glosses'] +
                              dataset['test']['pos_glosses'] for gloss in glosses.replace("-", " ").split()]))
    all_glosses = ["<sep>", "<pad>"] + all_glosses
    SEP_TOKEN_ID = all_glosses.index("<sep>")
    PAD_TOKEN_ID = all_glosses.index("<pad>")
    print(f"{len(all_glosses)} unique glosses")

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
                                       feature_lengths=[len(all_glosses)],
                                       feature_map=[[i] for i in range(len(all_glosses))],
                                       decoder_pad_token_id=PAD_TOKEN_ID,
                                       decoder_max_length=MAX_OUTPUT_LENGTH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Training loop
    for epoch in tqdm(range(MAX_EPOCHS), desc="Epoch", colour="blue"):
        start_time = timer()

        model.train()
        losses = 0

        for batch in tqdm(dataloader, desc="Batch", colour="green"):
            (loss, feature_logits) = model.forward(input_ids=batch['input_ids'].to(device),
                                                   attention_mask=batch['attention_mask'].to(device),
                                                   decoder_input_ids=batch['labels'].to(device),
                                                   features=batch['labels'].to(device).unsqueeze(-2))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            losses += loss.item()

        scheduler.step()
        end_time = timer()
        train_loss = losses / len(dataloader)

        # Eval
        eval_loss, metrics = eval_epoch(model, eval_dataloader, all_glosses)

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

    # Run final eval on test
    test_preds, test_metrics = test(model, test_dataloader, all_glosses)
    wandb.log({
        "test": test_metrics
    })
    with open('preds.out', 'w') as f:
        for seq in test_preds:
            # Join the strings in the inner list with a space and write to the file
            f.write(' '.join(seq) + '\n')

    wandb.finish()
    torch.save(model.state_dict(), "./model.pth")


if __name__ == "__main__":
    fire.Fire(main)
