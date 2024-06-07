from llmtuner import run_exp
import deepspeed
deepspeed.ops.op_builder.CPUAdamBuilder().load()

def main():
    run_exp()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
