

import pandas as pd
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import DistilBertConfig, DistilBertForSequenceClassification, PreTrainedTokenizerFast

import finetune as ft
import make_dataset
import prepare_data as prep

def test_clean_text_removes_noise_and_fixes_typos():
    assert prep.clean_text("  <p>I forgot my PASWORD</p>   ") == "i forgot my password"
    assert prep.clean_text("Resrvation was cancellled") == "reservation was cancelled"
    assert prep.clean_text("please cancell it") == "please cancel it"
    assert prep.clean_text("cancelled stay") == "cancelled stay"  # правильное слово не портится


def test_clean_dataframe_drops_bad_rows_and_duplicates():
    raw = pd.DataFrame({
        "text": ["My card was declined", "  MY CARD WAS DECLINED ", "", "Hello?", "Reset my pasword please"],
        "label": [" Payment ", "payment", "booking", "", "ACCOUNT"],
    })
    clean, report = prep.clean_dataframe(raw)
    assert clean.to_dict("records") == [
        {"text": "my card was declined", "label": "payment"},
        {"text": "reset my password please", "label": "account"},
    ]
    assert report["raw_rows"] == 5 and report["after_dedup"] == 2


def test_full_raw_dataset_cleans_back_to_150_unique_tickets(tmp_path, monkeypatch):
    monkeypatch.setattr(make_dataset, "DATA_DIR", tmp_path)
    raw = pd.read_csv(make_dataset.main(), keep_default_na=False)
    clean, _ = prep.clean_dataframe(raw)
    assert len(clean) == 150
    train, test = prep.split(clean)
    assert (len(train), len(test)) == (120, 30)
    assert set(train["text"]).isdisjoint(test["text"])
    assert test["label"].value_counts().to_dict() == {l: 6 for l in prep.LABELS}


@pytest.fixture(scope="module")
def splits(tmp_path_factory):
    from make_dataset import TICKETS
    rows = [(t.lower(), l) for l, ts in TICKETS.items() for t in ts]
    df = pd.DataFrame(rows, columns=["text", "label"])
    return prep.split(df)


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory, splits):
    path = tmp_path_factory.mktemp("tiny_distilbert")
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(splits[0]["text"], trainers.WordLevelTrainer(
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="[PAD]", unk_token="[UNK]",
                                     cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    hf_tok.save_pretrained(path)
    config = DistilBertConfig(vocab_size=hf_tok.vocab_size, dim=32, hidden_dim=64,
                              n_layers=1, n_heads=2, max_position_embeddings=64,
                              num_labels=len(prep.LABELS))
    DistilBertForSequenceClassification(config).save_pretrained(path)
    return path


def test_baseline_is_better_than_random(splits):
    result = ft.run_baseline(*splits)
    assert result["accuracy"] > 0.5  # случайное угадывание = 0.2


def test_finetune_evaluate_and_predict_end_to_end(splits, tiny_model_dir, tmp_path):
    train, test = splits
    out = tmp_path / "model"
    trainer = ft.finetune(train, test, model_name=str(tiny_model_dir), output_dir=out, epochs=1)

    result = ft.evaluate(trainer, test)
    assert set(result) == {"accuracy", "f1_macro"}
    assert 0 <= result["accuracy"] <= 1

    assert (out / "config.json").exists()
    labels = ft.predict(["I forgot my password", "Refund please"], model_dir=out)
    assert len(labels) == 2 and set(labels) <= set(prep.LABELS)
