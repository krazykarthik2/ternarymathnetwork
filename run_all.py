"""
Master test runner: Executes Objectives 1 through 5 in sequence.
Collects all empirical proof logs, tables, metrics, and generates publication plots in results/.
"""
import os
import sys
import time
from obj1_staircase_paradox import run_objective_1
from obj2_nascent_delta import run_objective_2
from obj3_ste_shadow_error import run_objective_3
from obj4_mige_gradient import run_objective_4
from obj5_training_comparison import run_objective_5

def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    start_total = time.time()
    print("\n" + "#" * 80)
    print("STARTING FULL SUITE: THE STAIRCASE PARADOX OF QUANTIZATION PROOFS")
    print("#" * 80 + "\n")

    # 1. Objective 1
    t0 = time.time()
    run_objective_1(results_dir)
    print(f">> Objective 1 completed in {time.time() - t0:.2f}s\n")

    # 2. Objective 2
    t0 = time.time()
    run_objective_2(results_dir)
    print(f">> Objective 2 completed in {time.time() - t0:.2f}s\n")

    # 3. Objective 3
    t0 = time.time()
    run_objective_3(results_dir)
    print(f">> Objective 3 completed in {time.time() - t0:.2f}s\n")

    # 4. Objective 4
    t0 = time.time()
    run_objective_4(results_dir)
    print(f">> Objective 4 completed in {time.time() - t0:.2f}s\n")

    # 5. Objective 5
    t0 = time.time()
    run_objective_5(results_dir, num_steps=600)
    print(f">> Objective 5 completed in {time.time() - t0:.2f}s\n")

    print("#" * 80)
    print(f"ALL OBJECTIVES VERIFIED AND EXECUTED. TOTAL TIME: {time.time() - start_total:.2f}s")
    print(f"Results stored in: {results_dir}")
    print("#" * 80 + "\n")

if __name__ == "__main__":
    main()
