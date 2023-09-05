import argparse
import os
import logging
import time
from tqdm import tqdm

from src.constants import MODELS, TASKS, FORMATS
from src.models import get_model
from src.prompts import generation_prompt

if __name__ == '__main__':
    if not os.path.exists('logs'):
        os.makedirs('logs')
    FORMAT = '[%(asctime)s] %(message)s'
    logging.basicConfig(format=FORMAT, filename=f'logs/generation_{time.strftime("%Y%m%d-%H%M%S")}.log', level=logging.INFO,
                        filemode='a', datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(
        prog='LLM Output Parsing Evaluation',
        description='This is a simple program for evaluating the output of the LLM models for a specific format and '
                    'task.')

    parser.add_argument("-m", "--model", type=str, required=True, nargs="+",
                        help='The model to be evaluated. Use "all" to evaluate all models.',
                        choices=MODELS + ['all'], )
    parser.add_argument("-t", "--task", type=str, required=True, nargs="+",
                        help='The task to be evaluated. Use "all" to evaluate all tasks.',
                        choices=TASKS + ['all'], )
    parser.add_argument("-f", "--format", type=str, required=True, nargs="+",
                        help='The format to be evaluated. Use "all" to evaluate all formats.',
                        choices=FORMATS + ['all'], )
    parser.add_argument('-n', "--trial", type=int, default=100,
                        help='The trial number of the model to be evaluated.')

    args = parser.parse_args()

    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    if args.model == ['all']:
        args.model = MODELS

    if args.task == ['all']:
        args.task = TASKS

    if args.format == ['all']:
        args.format = FORMATS

    logging.info(f"Starting generation for {args.model} on {args.task} with {args.format}")
    for task in tqdm(args.task, desc="Task", position=0):
        if not os.path.exists(f'outputs/{task}'):
            os.makedirs(f'outputs/{task}')

        for model in tqdm(args.model, desc="Model", position=1):
            if not os.path.exists(f'outputs/{task}/{model}'):
                os.makedirs(f'outputs/{task}/{model}')

            for output_format in tqdm(args.format, desc="Format", position=2):
                if not os.path.exists(f'outputs/{task}/{model}/{output_format}'):
                    os.makedirs(f'outputs/{task}/{model}/{output_format}')

                prompt = generation_prompt(task, output_format)
                try:
                    chat_model = get_model(model)

                    finished_files = len([name for name in os.listdir(f'outputs/{task}/{model}/{output_format}') if
                                          os.path.isfile(
                                              os.path.join(f'outputs/{task}/{model}/{output_format}', name))])

                    for i in tqdm(range(finished_files, args.trial), desc="Trial", position=3):
                        logging.info(f"Starting trial {i} for {model} on {task} with {output_format}")
                        response = chat_model.inference(prompt)

                        with open(f'outputs/{task}/{model}/{output_format}/{i + 1}.txt', 'w') as f:
                            f.write(response)

                        logging.info(f"Finished trial {i} for {model} on {task} with {output_format}")
                except ValueError:
                    print(f"Model {model} not found.")
                    continue
