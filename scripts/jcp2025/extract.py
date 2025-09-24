import json
import pickle
from pathlib import Path
from typing import Union, Callable

from runner import Runner


def get_jsonl_data(path: Union[str, Path], get_data: Callable):
    """"""
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            # Get results from file
            line_data = json.loads(line)

            # Run getter function
            result = get_data(line_data)
            if result[0] == True:
                data.append(
                    {
                        key: line_data[key]
                        for key in [
                            "num_ordinates",
                            "num_groups",
                            "factor",
                            "degree",
                            "eps",
                            "device",
                        ]
                    }.update(result[1])
                )

    return data


def get_pickle_data(dir: Union[str, Path], case: dict):
    """"""
    with open(
        Path(dir)
        / "N{}_G{}_A{}_B{}_p{}_q{}_eps{}{}".format(
            *Runner.case_name(
                case["num_ordinates"],
                case["num_groups"],
                case["factor"],
                case["degree"],
                case["eps"],
                case["device"],
            )[:-1],
            case["device"]
        ),
        "rb",
    ) as f:
        return pickle.load(f)
