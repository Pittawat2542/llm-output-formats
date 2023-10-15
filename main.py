import argparse
import json
import os
import logging
import time
import xml

import yaml
from tqdm import tqdm

from src.constants import MODELS, TASKS, FORMATS, OUTPUT_FOLDER, PARSED_OUTPUT_FOLDER
from src.models import get_model
from src.parsers import parse_json, parse_xml, parse_yaml
from src.prompts import generation_prompt
from src.types import Task, OutputFormat


def generation_task(models: list[str], tasks: Task, output_formats: OutputFormat, trial: int):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    logging.info(f"Starting generation for {models} on {tasks} with {format}")
    for task in tqdm(tasks, desc="Task", position=0):
        if not os.path.exists(f'{OUTPUT_FOLDER}/{task}'):
            os.makedirs(f'{OUTPUT_FOLDER}/{task}')

        for model in tqdm(models, desc="Model", position=1):
            if not os.path.exists(f'{OUTPUT_FOLDER}/{task}/{model}'):
                os.makedirs(f'{OUTPUT_FOLDER}/{task}/{model}')

            for output_format in tqdm(output_formats, desc="Format", position=2):
                if not os.path.exists(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}'):
                    os.makedirs(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}')

                prompt = generation_prompt(task, output_format)
                try:
                    chat_model = get_model(model)

                    finished_files = len(
                        [name for name in os.listdir(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}') if
                         os.path.isfile(
                             os.path.join(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}', name))])

                    for i in tqdm(range(finished_files, trial), desc="Trial", position=3):
                        logging.info(f"Starting trial {i} for {model} on {task} with {output_format}")
                        response = chat_model.inference(prompt)

                        with open(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}/{i + 1}.txt', 'w') as f:
                            if response is None:
                                f.write("")
                            else:
                                f.write(response)

                        logging.info(f"Finished trial {i} for {model} on {task} with {output_format}")
                except ValueError as e:
                    print(f"Unexpected error: {e}")
                    continue


def parsing_task():
    if not os.path.exists(PARSED_OUTPUT_FOLDER):
        os.makedirs(PARSED_OUTPUT_FOLDER)

    logging.info("Starting parsing")
    for task in tqdm(os.listdir(OUTPUT_FOLDER), desc="Task", position=0):
        if not os.path.exists(f'{PARSED_OUTPUT_FOLDER}/{task}'):
            os.makedirs(f'{PARSED_OUTPUT_FOLDER}/{task}')

        for model in tqdm(os.listdir(f'{OUTPUT_FOLDER}/{task}'), desc="Model", position=1):
            if not os.path.exists(f'{PARSED_OUTPUT_FOLDER}/{task}/{model}'):
                os.makedirs(f'{PARSED_OUTPUT_FOLDER}/{task}/{model}')

            for output_format in tqdm(os.listdir(f'{OUTPUT_FOLDER}/{task}/{model}'), desc="Format", position=2):
                if not os.path.exists(f'{PARSED_OUTPUT_FOLDER}/{task}/{model}/{output_format}'):
                    os.makedirs(f'{PARSED_OUTPUT_FOLDER}/{task}/{model}/{output_format}')

                for trial in tqdm(os.listdir(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}'), desc="Trial",
                                  position=3):
                    with open(f'{OUTPUT_FOLDER}/{task}/{model}/{output_format}/{trial}', 'r') as f:
                        logging.info(f"Starting parsing {task} {model} {output_format} {trial}")
                        try:
                            output = f.read()
                        except UnicodeDecodeError as e:
                            logging.error(f"Error parsing {task} {model} {output_format} {trial}: {e}")
                            with open(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json', 'r+') as f2:
                                summarized_results = json.load(f2)

                                summarized_results["results"].append({
                                    "task": task,
                                    "model": model,
                                    "output_format": output_format,
                                    "trial": trial,
                                    "error": str(e)
                                })

                                f2.seek(0)
                                f2.truncate()
                                f2.write(json.dumps(summarized_results, indent=2))
                            continue

                        try:
                            match output_format:
                                case "json":
                                    result = parse_json(output)
                                case "xml":
                                    result = parse_xml(output)
                                case "yaml":
                                    result = parse_yaml(output)
                                case _:
                                    raise ValueError(f"Unexpected output format: {output_format}")

                            if result is None:
                                logging.error(f"Error parsing {task} {model} {output_format} {trial}")
                            else:
                                with open(f'{PARSED_OUTPUT_FOLDER}/{task}/{model}/{output_format}/{trial}.json',
                                          'w') as f:
                                    f.write(json.dumps(result, indent=2))

                                if not os.path.exists(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json'):
                                    with open(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json', 'w') as f:
                                        f.write(json.dumps({
                                            "results": []
                                        }))

                                with open(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json', 'r+') as f2:
                                    summarized_results = json.load(f2)

                                    summarized_results["results"].append({
                                        "task": task,
                                        "model": model,
                                        "output_format": output_format,
                                        "trial": trial,
                                        "result": result
                                    })

                                    f2.seek(0)
                                    f2.truncate()
                                    f2.write(json.dumps(summarized_results, indent=2))

                                logging.info(f"Finished parsing {task} {model} {output_format} {trial}")
                        except (json.decoder.JSONDecodeError, xml.parsers.expat.ExpatError, yaml.scanner.ScannerError,
                                yaml.parser.ParserError, yaml.composer.ComposerError,
                                yaml.constructor.ConstructorError) as e:
                            if not os.path.exists(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json'):
                                with open(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json', 'w') as f:
                                    f.write(json.dumps({
                                        "results": []
                                    }))

                            with open(f'{PARSED_OUTPUT_FOLDER}/summarized_results.json', 'r+') as f2:
                                summarized_results = json.load(f2)

                                summarized_results["results"].append({
                                    "task": task,
                                    "model": model,
                                    "output_format": output_format,
                                    "trial": trial,
                                    "error": str(e)
                                })

                                f2.seek(0)
                                f2.truncate()
                                f2.write(json.dumps(summarized_results, indent=2))
                            logging.error(f"Error parsing {task} {model} {output_format} {trial}: {e}")


if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')

    parser = argparse.ArgumentParser(
        prog='LLM Output Parsing Evaluation',
        description='This is a simple program for evaluating the output of the LLM models for a specific format and '
                    'task.')

    parser.add_argument("--mode", type=str, default="generation", choices=["generation", "parsing"],
                        help='The mode of the program. "generation" for generating outputs and "parsing" for parsing ')
    parser.add_argument("-m", "--model", type=str, nargs="+",
                        help='The model to be evaluated. Use "all" to evaluate all models.',
                        choices=MODELS + ['all'], )
    parser.add_argument("-t", "--task", type=str, nargs="+",
                        help='The task to be evaluated. Use "all" to evaluate all tasks.',
                        choices=TASKS + ['all'], )
    parser.add_argument("-f", "--format", type=str, nargs="+",
                        help='The format to be evaluated. Use "all" to evaluate all formats.',
                        choices=FORMATS + ['all'], )
    parser.add_argument('-n', "--trial", type=int, default=100,
                        help='The trial number of the model to be evaluated.')

    args = parser.parse_args()

    FORMAT = '[%(asctime)s] %(message)s'
    logging.basicConfig(format=FORMAT, filename=f'logs/{args.mode}_{time.strftime("%Y%m%d-%H%M%S")}.log',
                        level=logging.INFO,
                        filemode='a', datefmt='%Y-%m-%d %H:%M:%S')

    if args.mode == "generation":
        if args.model == ['all']:
            args.model = MODELS

        if args.task == ['all']:
            args.task = TASKS

        if args.format == ['all']:
            args.format = FORMATS

        generation_task(args.model, args.task, args.format, args.trial)
    elif args.mode == "parsing":
        parsing_task()
