from demo_rq_simlab_scenarios_gen_parametric_normal import (
    run_generator_demo as run_parametric_normal_demo,
)
from demo_rq_simlab_scenarios_gen_parametric_students_t import (
    run_generator_demo as run_parametric_student_t_demo,
)
from demo_rq_simlab_scenarios_gen_parametric_gmm import (
    run_generator_demo as run_parametric_gmm_demo,
)
from demo_rq_simlab_scenarios_gen_historical import (
    run_generator_demo as run_historical_demo,
)
from demo_rq_simlab_scenarios_gen_historical_filtered import (
    run_generator_demo as run_historical_filtered_demo,
)
from demo_rq_simlab_scenarios_gen_bootstrap import (
    run_generator_demo as run_bootstrap_demo,
)
from demo_rq_simlab_scenarios_gen_bootstrap_filtered import (
    run_generator_demo as run_bootstrap_filtered_demo,
)


if __name__ == "__main__":
    print("\n===============================================================")
    print("=== Running Parametric Normal Scenario Generator Demo       ===")
    print("===============================================================\n")
    run_parametric_normal_demo()
    print("\n===============================================================")
    print("=== Running Parametric Student's t Scenario Generator Demo  ===")
    print("===============================================================\n")
    run_parametric_student_t_demo()
    print("\n===============================================================")
    print("=== Running Parametric GMM Scenario Generator Demo          ===")
    print("===============================================================\n")
    run_parametric_gmm_demo()
    print("\n===============================================================")
    print("=== Running Historical Scenario Generator Demo              ===")
    print("===============================================================\n")
    run_historical_demo()
    print("\n===============================================================")
    print("=== Running Historical Filtered Scenario Generator Demo     ===")
    print("===============================================================\n")
    run_historical_filtered_demo()
    print("\n===============================================================")
    print("=== Running Bootstrap Scenario Generator Demo      ===")
    print("===============================================================\n")
    run_bootstrap_demo()
    print("\n===============================================================")
    print("=== Running Bootstrap Filtered Scenario Generator Demo      ===")
    print("===============================================================\n")
    run_bootstrap_filtered_demo()
