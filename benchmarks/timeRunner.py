#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Varun Nayyar <nayyarv@gmail.com>"

import timeit
import click

N = 1000


setup = """
import torch

N = {N}
device = "{device}"

a = torch.rand(N, N, dtype=torch.float32, device=device)
b = torch.rand(N, N, dtype=torch.float32, device=device)
"""

runs = "a @ b"

@click.command()
@click.option("--method",
              type=click.Choice(["cpu", "cuda"]))
@click.option("--reps", default=100)
def main(method, reps):
    print(f"{method} ({reps} iterations)")
    print("# pow, N, tot_time(s), scaled_time (us)")
    for Npow in range(2, 5):
        N = 10 ** Npow
        rtime = timeit.timeit(runs, setup.format(N=N, device=method),
                              number=reps)
        print(f"{Npow}, {N}, {rtime:.2f}, {rtime/N * 10**4:.2f}")
    print()


if __name__ == '__main__':
    main()
