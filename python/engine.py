import os
import argparse
from generate import CloudGenerator
from simulate import Simulator
from visualize import PostProcessor
from mpi4py import MPI

os.system('clear')
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config input file path')
    parser.add_argument('--generate', action='store_true', help='generate initial configuration')
    parser.add_argument('--simulate', action='store_true', help='perform the simulation')
    parser.add_argument('--visualize', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if args.config == None:
        if rank == 0:
            print('Please input config file')
    else:
        if rank == 0:
            print(f'Parsing config file: {args.config}')
    
    if args.generate:
        if rank == 0:
            generator = CloudGenerator(args.config)
    
    if args.simulate:
        simulator = Simulator(args.config)
        
        simulator.run()
    
    if args.visualize:
        if rank == 0:
            post_processor = PostProcessor(args.config)