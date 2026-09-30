import json
from pathlib import Path
from metrics import binary_metrics

def main():
    cases=json.loads(Path(__file__).with_name("cases.json").read_text())
    # This runner consumes recorded predictions so it can be reused by model adapters.
    predictions=[c.get("prediction","unknown") for c in cases]
    expected=[c["expected"] for c in cases]
    m=binary_metrics(expected,predictions)
    print(json.dumps({"n":len(cases),"precision":m.precision,"recall":m.recall,"fnr":m.false_negative_rate,"fpr":m.false_positive_rate},indent=2))
if __name__=="__main__": main()
