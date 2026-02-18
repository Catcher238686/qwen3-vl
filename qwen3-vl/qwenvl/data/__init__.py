import re

YOUR_CAPTION_DATASET = {
    "annotation_path": "/path/to/your_captions.jsonl",
    "data_path": "",
}

RSCC_DATASET = {
    "annotation_path": "/path/to/rscc_qwenvl_format.jsonl",
    "data_path": "",
}

RSCC_SUBSET_DATASET = {
    "annotation_path": "/path/to/rscc_subset_qwenvl_format.jsonl",
    "data_path": "",
}

data_dict = {
    "your_caption_dataset": YOUR_CAPTION_DATASET,
    "rscc_subset": RSCC_SUBSET_DATASET,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["your_caption_dataset%100"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
        import os

        if not os.path.exists(config["annotation_path"]):
            raise FileNotFoundError(
                f"Annotation file not found: {config['annotation_path']}"
            )
