'''
This script adds log information to the test assertions. When an assertion fails, it will now provide details about the input and expected/actual output.
'''

import json
import re
import gzip
import os


if __name__ == "__main__":

    root_dir = "/evo/homes/crupig/benchmarks/MultiPL-E/generations"

    # walk through all files in the root_dir
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".json.gz"):
                file_path = os.path.join(subdir, file)

            with gzip.open(file_path, "rt") as f:
                log_data = json.load(f)

            tests = log_data["tests"]

            patterns = [r"assert\s*\(\s*(\w+)\s*\((.*)\)\.equals\s*\((.*)\)\s*\)\s*", r"assert\s*\(\s*(\w+)\s*\((.*)\)\s*==\s*(.*)\s*\)\s*;"]
            new_tests = []

            for line in tests.splitlines():
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match and "For input" not in line:
                        function_name, input_value, expected_output = match.groups()
                        assert line.endswith(";")
                        new_tests.append(line[:-1] + f" : \"For input {input_value.replace("\"", "")}: expected {expected_output.replace("\"", "")}, but got \" + {function_name}({input_value});")
                        break
                    
                    if pattern == patterns[-1]:
                        new_tests.append(line)
                    
            tests = "\n".join(new_tests)

            log_data["tests"] = tests

            # overwrite the log file with the updated tests
            with gzip.open(file_path, "wt") as f:
                json.dump(log_data, f, indent=4)