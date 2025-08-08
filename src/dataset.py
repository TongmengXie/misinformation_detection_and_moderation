from datasets import load_dataset

LABELS = ["pants-fire", "false", "barely-true", "half-true", "mostly-true", "true"]

def get_liar(split='train', binarize=True):
    ds = load_dataset("chengxuphd/liar2", split=split)
    def add_label_name(x):
        i = int(x["label"])
        x["label_name"] = LABELS[i] if 0 <= i < len(LABELS) else "unknown"
        return x
    ds = ds.map(add_label_name)
    if binarize:
        true_set = {"half-true", "mostly-true", "true"}
        def binarize_fn(x):
            x["label_bin"] = "true" if x["label_name"] in true_set else "false"
            return x
        ds = ds.map(binarize_fn)
    return ds
