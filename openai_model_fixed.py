"""
Use to get completions from an OpenAI completions endpoint. This version
of the script only works with Azure OpenAI service, since OpenAI no longer
hosts their code completion models.
"""
from typing import List
from multipl_e.completions_gpt import partial_arg_parser, make_main
import openai
import time
from typing import List
import re

global engine, model

client = openai.OpenAI()


def completions(
    prompts: List[str], max_tokens: int, temperature: float, top_p, stop
) -> List[str]:
    results = []
    for prompt in prompts:
        kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            # "stop": stop
        }
        if engine is not None:
            kwargs["engine"] = engine
        elif model is not None:
            kwargs["model"] = model

        while True:
            attempt = 0
            if attempt > 5:
                return ""
            try:                
                result = client.chat.completions.create(messages=[
                    # {"role": "system", "content": "You are a helpful assistant for writing Java code. You will be given a Java method description and its signature, and you will write the whole method. Only return the code for the method, and no other text."},
                    {"role": "user", "content": prompt}
                    ], **kwargs)
                result = result.choices[0].message.content
                
                # output postprocess
                pattern = r'```java(.*?)```'
                match = re.search(pattern, result, re.DOTALL)
                if match:
                    result = match.group(1).strip()
                    result = '\n'.join([line for line in result.split('\n')[1:-1]])

                    
                break
            except Exception as e:
                print("Exception:", e)
                time.sleep(5)
                attempt += 1
        results.append(result)

        # Sleep to avoid rate limiting
        time.sleep(0.5)
    return results



def main():
    global engine, model
    args = partial_arg_parser()
    args.add_argument("--model", type=str)
    args.add_argument("--engine", type=str)
    args.add_argument("--name-override", type=str)
    args.add_argument("--azure", action="store_true")
    args.add_argument("--all_ids_dict", default=None, type=str)
    args.add_argument("--split", choices=["train", "test", "val", "all"], type=str, help="Subset of the data to run on (train/val/test/all).")
    args = args.parse_args()

    if args.engine is None and args.model is None:
        raise ValueError("Must specify either engine or model.")
    elif args.engine is not None and args.model is not None:
        raise ValueError("Must specify either engine or model, not both.")

    engine = args.engine
    model = args.model
    if args.name_override:
        name = args.name_override
    else:
        if args.engine is not None:
            name = args.engine
        else:
            name = args.model

    make_main(args, name, completions)


if __name__ == "__main__":
    main()