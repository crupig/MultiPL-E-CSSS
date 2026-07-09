import os
import json
import pandas as pd
import sys
import gzip
from tqdm import tqdm

def write_for_test_execution(row):
    test_id = row['test_idx'].split('--TestID::')[1]
    sample_id = row['solution_idx'][0].split('--SampleID::')[0]
    
    output_dir = os.path.join("../test-4-execution", row['generated_by'], row['task_idx'])
    if os.path.exists(os.path.join(output_dir, f"{sample_id}--TestID::{test_id}.json.gz")):
        pass
    
    test_statement = row['test_statement'].replace("Problem.", "")
    test_code = "    }\n\n    public static void main(String[] args) {\n    " + test_statement + "\n}\n\n}\n"
    completions = []
    for method in row['method']:
        completion = '\n'.join([line for line in method.strip().splitlines()[1:-1]])
        completions.append(completion)

    assert len(completions) == len(row['solution_idx']), f"Number of completions {len(completions)} does not match number of solution idx {len(row['solution_idx'])}"
    data = {
        "name": row['task_idx'],
        # "test_execution_idx": f"{row['onwhichtomerge']}--TestID::{test_id}--SampleID::{sample_id}",
        "test_idx": row['test_idx'],
        "solution_idx": row['solution_idx'],
        "language": "java",
        "generated_by": row['generated_by'],
        "prompt": row['prompt'],
        "test_statement": test_statement,
        "tests": test_code,
        "completions": completions,
        "stop_tokens": [
            "\n    }\n"
        ]
    }

    os.makedirs(output_dir, exist_ok=True)
    try:
        with gzip.open(os.path.join(output_dir, f"{sample_id}--TestID::{test_id}.json.gz"), "wt") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"NEGATIVO: {e}")

if __name__ == "__main__":
    tqdm.pandas()
    
    # import knowledge base
    knowlbase_path = "./MultiPL-E/knowlbase/knowlbase_multiple.json"
    with open(knowlbase_path, "r") as f:
        kb = json.load(f)
    
    kb = pd.DataFrame(kb)
    kb = kb[['solution_idx', 'task_idx', 'generated_by', 'prompt', 'method']]
    kb['onwhichtomerge'] = kb['solution_idx'].apply(lambda x: x.split("--SampleID")[0])

    
    # FILTER KNOWLEDGE BASE TO TEST SET SOLUTIONS
    with open("../data/finetuning/test_upto20_real.jsonl", "r") as f:
        test_set_uptp20 = [json.loads(line) for line in f]
    test_set_uptp20 = pd.DataFrame(test_set_uptp20)
    solution_ids_to_keep = test_set_uptp20.solution_idx.unique().tolist()
    kb = kb[kb.solution_idx.isin(solution_ids_to_keep)]
    del test_set_uptp20
    del solution_ids_to_keep
    
    
    # SPLIT BY GENERATED_BY
    for generated_by in kb['generated_by'].unique():
        # IMPORT TESTS KNOWLEDGE BASE
        test_path = f"../knowlbase-tests/{generated_by}_knowlbase_tests_multiple.jsonl"
        if not os.path.exists(test_path):
            continue
        
        try:
            kbt = pd.read_json(test_path, lines=True)
        except Exception as e:
            print(f"Check for possible inconsistencies in {generated_by}: ", e)

        print(f"################## {generated_by} ##################")
        num_task_ids = json.load(open('../../../constants/ids_train_val_test.json', 'r'))['MultiPL-E']['test']
        iii = kb.loc[(kb['generated_by'] == generated_by) & (kb['task_idx'].isin(num_task_ids)), 'task_idx'].unique()
        print(f"RRRR:\t\t{len(iii)} / {len(num_task_ids)}")
        print(f"RRRR:\t\t {set(num_task_ids)-set(iii)}")
        # exit()
        kbt['onwhichtomerge'] = kbt['test_idx'].apply(lambda x: x.split("--TestID")[0])
        print(f"RRR {kbt['task_idx'].nunique()} / {len(num_task_ids)}")
        kbt = kbt.drop(columns=['task_idx', 'generated_by', 'num_unique_asserts'])

        merged = kb.merge(kbt, on="onwhichtomerge", how="inner")

        ###############
        g = merged.groupby('solution_idx', as_index=False).agg({'test_idx': 'count', 'task_idx': 'first'})
        print(f"################## {generated_by} ##################")
        print(f"Max number of unique asserts per task ID:\t\t{g['test_idx'].max()}")
        print(f"Min number of unique asserts per task ID:\t\t{g['test_idx'].min()}")
        print(f"Average number of unique asserts per task ID:\t\t{g['test_idx'].mean():.2f}")
        print(f"Median number of unique asserts per task ID:\t\t{g['test_idx'].median()}")
        print(f"Number of task IDs with no asserts (out of {len(num_task_ids)}):\t\t{len(num_task_ids) - g['task_idx'].nunique()}")
        print()
        ###############

        g = merged.groupby('test_idx', as_index=False).agg({
            'task_idx' : 'first',
            'solution_idx' : list,
            'method' : list,
            'test_statement' : 'first',
            'generated_by' : 'first',
            'prompt' : 'first',
        })

        authorized = input(f"Do you want to save the tests extracted for {generated_by} to '../test-4-execution'? (y/n): ")
        if authorized.lower() in ['y', 'yes']:
            g.progress_apply(write_for_test_execution, axis=1)
        else:
            print("Skipped.")