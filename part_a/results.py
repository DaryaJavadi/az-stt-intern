import os
import pandas as pd
import evaluate


def generate_results(preds, refs):

    #metrics:
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")

    wer_score = wer.compute(
        predictions=preds,
        references=refs
    )

    cer_score = cer.compute(
        predictions=preds,
        references=refs
    )

    print("\nFinal results:")
    print("WER:", wer_score)
    print("CER:", cer_score)

    # Reporting per sample:
    results = pd.DataFrame({
        "reference": refs,
        "prediction": preds
    })

    results["wer"] = results.apply(
        lambda x: wer.compute(
            predictions=[x["prediction"]],
            references=[x["reference"]]
        ),
        axis=1
    )

    results["cer"] = results.apply(
        lambda x: cer.compute(
            predictions=[x["prediction"]],
            references=[x["reference"]]
        ),
        axis=1
    )

    # Final results:
    avg_wer = results["wer"].mean() * 100
    avg_cer = results["cer"].mean() * 100

    print("\nOverall metrics:")
    print(f"Average WER: {avg_wer:.2f}%")
    print(f"Average CER: {avg_cer:.2f}%")

    # Best 5 samples:
    best_5 = results.sort_values("wer").head(5)

    print("\nBest 5 samples:")

    for _, row in best_5.iterrows():

        print("\nReference:", row["reference"])
        print("Prediction:", row["prediction"])
        print("WER:", round(row["wer"], 3))
        print("CER:", round(row["cer"], 3))
        print("-" * 50)

    # Worst 5 samples:
    worst_5 = results.sort_values(
        "wer",
        ascending=False
    ).head(5)

    print("\nWorst 5 samples:")

    for _, row in worst_5.iterrows():

        print("\nReference:", row["reference"])
        print("Prediction:", row["prediction"])
        print("WER:", round(row["wer"], 3))
        print("CER:", round(row["cer"], 3))
        print("-" * 50)

    # Saving the results:
    os.makedirs("results", exist_ok=True)

    results.to_csv(
        "results/part_a_results.csv",
        index=False
    )

    print("\nSaved -> results/part_a_results.csv")