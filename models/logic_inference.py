import json
import os
from tqdm import tqdm
from symbolic_solvers.fol_solver.prover9_solver import FOL_Prover9_Program
from symbolic_solvers.pyke_solver.pyke_solver import Pyke_Program
from symbolic_solvers.z3_solver.sat_problem_solver import LSAT_Z3_Program
import argparse
from collections import defaultdict
import time
from backup_answer_generation import Backup_Answer_Generator


class LogicInferenceEngine:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.backup_strategy = args.backup_strategy
        self.solver_mode = args.solver_mode

        self.dataset = self.load_logic_programs()
        program_executor_map = {
            "FOLIO": FOL_Prover9_Program,
            "ProntoQA": Pyke_Program,
            "ProofWriter": Pyke_Program,
            "LogicalDeduction": LSAT_Z3_Program,
            "AR-LSAT": LSAT_Z3_Program,
        }
        self.program_executor = program_executor_map[self.dataset_name]
        self.backup_generator = Backup_Answer_Generator(
            self.dataset_name, self.backup_strategy, self.args.backup_LLM_result_path
        )

    def load_logic_programs(self):
        with open(
            os.path.join(
                "./outputs/logic_programs",
                f"{self.dataset_name}_{self.split}_{self.model_name}.json",
            )
        ) as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} examples from {self.split} split.")
        return dataset

    def save_results(self, outputs):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(
            os.path.join(
                self.save_path,
                f"{self.dataset_name}_{self.split}_{self.model_name}_backup-{self.backup_strategy}_{self.solver_mode}.json",
            ),
            "w",
        ) as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

    def safe_execute_program(self, id, logic_program):
        program = self.program_executor(
            logic_program, self.dataset_name, mode=self.solver_mode
        )

        # 1. Parsing Check
        if program.flag == False:
            answer = self.backup_generator.get_backup_answer(id)
            return answer, "parsing error", "Program init failed", 0.0, {}

        # 2. Execution with Timing
        start_time = time.time()
        try:
            answer, error_message = program.execute_program()
        except Exception as e:
            answer = None
            error_message = str(e)
        end_time = time.time()
        duration = end_time - start_time

        # aggregate Solver Stats
        aggregated_stats = defaultdict(float)
        max_memory = 0.0

        if answer:
            lines = answer.splitlines() if isinstance(answer, str) else answer
            for line in lines:
                if line.strip().startswith("STATS:::"):
                    try:
                        json_str = line.strip().replace("STATS:::", "")
                        stats = json.loads(json_str)

                        # add every metric found
                        for k, v in stats.items():
                            if k == "memory":
                                max_memory = max(max_memory, float(v))
                            else:
                                aggregated_stats[k] += int(v)

                    except:
                        pass

        solver_stats = dict(aggregated_stats)
        solver_stats["memory"] = max_memory

        # 3. Handle Results
        if answer is None:
            # Distinguish between Timeout and Logic Error
            status = "timeout" if "Timeout" in error_message else "execution error"
            final_ans = self.backup_generator.get_backup_answer(id)
            return final_ans, status, error_message, duration, solver_stats

        # 4. Map Answer
        final_ans = program.answer_mapping(answer)
        if final_ans is None:
            # Solver ran but produced no valid option (A-E)
            final_ans = self.backup_generator.get_backup_answer(id)
            return (
                final_ans,
                "mapping error",
                "No valid option found in output",
                duration,
                solver_stats,
            )

        return final_ans, "success", "", duration, solver_stats

    def inference_on_dataset(self):
        outputs = []
        stats = {
            "success": 0,
            "parsing error": 0,
            "execution error": 0,
            "timeout": 0,
            "mapping error": 0,
        }

        for example in tqdm(self.dataset):
            answer, flag, error_message, duration, solver_stats = (
                self.safe_execute_program(
                    example["id"], example["raw_logic_programs"][0].strip()
                )
            )
            if flag in stats:
                stats[flag] += 1
            else:
                stats["execution error"] += 1
            output = {
                "id": example["id"],
                "context": example["context"],
                "question": example["question"],
                "answer": example["answer"],
                "flag": flag,
                "error_message": error_message,
                "inference_time": duration,
                "solver_stats": solver_stats,
                "predicted_answer": answer,
            }
            outputs.append(output)

        print(f"Execution Stats: {stats}")
        self.save_results(outputs)
        self.cleanup()

    def cleanup(self):
        complied_krb_dir = "./models/compiled_krb"
        if os.path.exists(complied_krb_dir):
            print("removing compiled_krb")
            os.system(f"rm -rf {complied_krb_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--save_path", type=str, default="./outputs/logic_inference")
    parser.add_argument(
        "--backup_strategy", type=str, default="random", choices=["random", "LLM"]
    )
    parser.add_argument(
        "--backup_LLM_result_path", type=str, default="../baselines/results"
    )
    parser.add_argument("--model_name", type=str, default="text-davinci-003")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--solver_mode",
        type=str,
        default="generic",
        choices=["generic", "theory_aware"],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    engine = LogicInferenceEngine(args)
    engine.inference_on_dataset()
