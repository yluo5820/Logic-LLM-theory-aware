# input: logic program file
# output: logic program file after one round of self-refinement

import json
import os
from tqdm import tqdm
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
import argparse
import random
from backup_answer_generation import Backup_Answer_Generator
from utils import OpenAIModel


class SelfRefinementEngine:
    def __init__(self, args, current_round):
        self.args = args
        self.split = args.split
        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.backup_strategy = args.backup_strategy
        self.openai_api = OpenAIModel(args.api_key, 'gpt-4', args.stop_words, args.max_new_tokens)
        self.current_round = current_round
        self.solver_mode = args.solver_mode

        self.logic_programs = self.load_logic_programs()
        # self.reasoning_results = self.load_inference_results()

        program_executor_map = {'AR-LSAT': LSAT_Z3_Program,
                                'FOLIO': FOL_Prover9_Program}
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path)

    def load_logic_programs(self):
        prefix = ""
        if self.current_round > 1:
            prefix = f'self-refine-{self.current_round-1}_'
        with open(os.path.join('./outputs/logic_programs', f'{prefix}{self.dataset_name}_{self.split}_{"gpt-4"}_{self.solver_mode}.json')) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset
    
    def load_prompt(self, program, error_message):
        program = program.strip()
        error_message = error_message.strip()
        with open(f'./models/prompts/self-correct-{self.dataset_name}.txt', 'r') as f:
            prompt_template = f.read()
        full_prompt = prompt_template.replace('[[PROGRAM]]', program).replace('[[ERROR MESSAGE]]', error_message)
        return full_prompt

    def safe_execute_program(self, id, logic_program, debug = False):
        program = self.program_executor(logic_program, self.dataset_name, mode=self.solver_mode)
        # cannot parse the program
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, 'parsing error', ''
        # execuate the program
        answer, error_message = program.execute_program()
        # not executable
        if answer is None:
            answer = self.backup_generator.get_backup_answer(id)

            ## output debug info
            if debug == True:
                if not os.path.exists('./debug'):
                    os.makedirs('./debug')
                with open(f'./debug/{id}.py', 'w') as f:
                    f.write(program.standard_code)
                with open(f'./debug/{id}.program.txt', 'w') as f:
                    f.write(logic_program)
                    f.write('\n')
                    f.write(error_message)

            return answer, 'execution error', error_message
        # successfully executed
        answer = program.answer_mapping(answer)
        return answer, 'success', error_message
    
    def single_round_self_refinement(self):
        """
        Run one round of self-refinement.

        - Uses the solver (with self.solver_mode) to execute each logic program.
        - If parsing/execution fails, calls the LLM to refine the program.
        - Writes progress to disk after each example.
        - If an output file for this round already exists, it resumes:
          examples whose IDs are already present are skipped.
        """
        # Where we save the current round
        save_path = f'./outputs/logic_programs/self-refine-{self.current_round}_{self.dataset_name}_{self.split}_{self.model_name}_{self.solver_mode}.json'

        # ---- 1. Load existing partial results (for resume) ----
        existing_outputs = []
        processed_ids = set()
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    existing_outputs = json.load(f)
                processed_ids = {ex["id"] for ex in existing_outputs}
                print(f"[SelfRefinement] Resuming round {self.current_round}: "
                      f"found {len(existing_outputs)} already-processed examples.")
            except Exception as e:
                print(f"[SelfRefinement] Warning: could not load existing file '{save_path}': {e}")
                existing_outputs = []
                processed_ids = set()

        # Fast lookup for existing entries by id
        existing_by_id = {ex["id"]: ex for ex in existing_outputs}

        # We'll rebuild outputs in dataset order, reusing existing entries when possible
        outputs = []
        for example in tqdm(self.logic_programs):
            ex_id = example['id']

            # ---- 2. If we've already processed this ID in this round, reuse it ----
            if ex_id in processed_ids:
                outputs.append(existing_by_id[ex_id])
                continue

            # ---- 3. Otherwise, run solver + (if needed) LLM refinement ----
            logic_program = example['raw_logic_programs'][0].strip()
            answer, status, error_message = self.safe_execute_program(ex_id, logic_program)

            if status == 'execution error':
                # execution error with some error message (real failure)
                if error_message != 'No Output':
                    full_prompt = self.load_prompt(logic_program, error_message)
                    revised_program = self.openai_api.generate(full_prompt).strip()
                    programs = [revised_program]
                    output = {
                        'id': example['id'],
                        'context': example['context'],
                        'question': example['question'],
                        'answer': example['answer'],
                        'options': example['options'],
                        'raw_logic_programs': programs,
                    }
                else:
                    # "No Output" is treated as non-fatal here - keep original
                    output = example

            elif status == 'parsing error':
                # parsing error: ask LLM to rewrite from scratch
                full_prompt = self.load_prompt(logic_program, 'Parsing Error')
                revised_program = self.openai_api.generate(full_prompt).strip()
                programs = [revised_program]
                output = {
                    'id': example['id'],
                    'context': example['context'],
                    'question': example['question'],
                    'answer': example['answer'],
                    'options': example['options'],
                    'raw_logic_programs': programs,
                }

            else:
                # status == 'success': keep original example untouched
                output = example

            # Append to in-memory list
            outputs.append(output)

            # ---- 4. Incremental save after EACH example ----
            # This ensures that if the script crashes (e.g., OpenAI 502),
            # all previous results are already on disk, and we can resume.
            with open(save_path, 'w') as f:
                json.dump(outputs, f, indent=2, ensure_ascii=False)

        # Done. The final file is already written by the last iteration.
        print(f"[SelfRefinement] Round {self.current_round} completed. "
              f"Saved {len(outputs)} examples to {save_path}.")

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maximum_rounds', type=int, default=3)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--backup_strategy', type=str, default='random', choices=['random', 'LLM'])
    parser.add_argument('--backup_LLM_result_path', type=str, default='../baselines/results')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--solver_mode', type=str, default='generic', choices=['generic', 'theory_aware'])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    for round in range(1, args.maximum_rounds+1):
        print(f"Round {round} self-refinement")
        engine = SelfRefinementEngine(args, round)
        engine.single_round_self_refinement()