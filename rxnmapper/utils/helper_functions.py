def find_input_type(file_path):
    if file_path.endswith("csv"):
        return "csv"
    elif file_path.endswith("txt"):
        return "txt"
    elif file_path.endswith("pkl"):
        return "pkl"
    elif file_path.endswith("json"):
        return "json"
    elif file_path.endswith("cxsmiles"):
        return "cxsmiles"
    else:
        raise ValueError(
            "Unsupported input file. Only .csv, .txt, .pkl, and .json files are supported"
        )
