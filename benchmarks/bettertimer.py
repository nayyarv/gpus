import inspect
import importlib
import timeit

import click

BENCHMARKS = ["nn", "matmul"]


def func_body(fun):
    """Code inspection magic"""
    data = [l.strip() for l in inspect.getsource(fun).split("\n")[1:]]
    return "\n".join(data)


@click.command()
@click.argument("bm", type=click.Choice(BENCHMARKS))
@click.argument("nvals", nargs=-1, type=int)
@click.option("--device",
              type=click.Choice(["cpu", "cuda"]))
@click.option("--reps", default=100)

def main(bm, nvals, device, reps):
    mod = importlib.import_module(bm)
    for N in nvals:
        
        setup = "\n".join([
            f"N={N}", 
            f"device='{device}'",
            func_body(mod.setup)])
        runner = func_body(mod.run)

        # print(setup)

        rtime = timeit.timeit(runner, setup, number=reps)
        print(f"N: {N:<5}; rtime: {rtime/reps:.5f}")

        




if __name__ == '__main__':
    main()