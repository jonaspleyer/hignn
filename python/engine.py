import os
import argparse
from generate import CloudGenerator
from simulate import Simulator
from visualize import PostProcessor

os.system('clear')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config input file path')
    parser.add_argument('--generate', action='store_true', help='generate initial configuration')
    parser.add_argument('--simulate', action='store_true', help='perform the simulation')
    parser.add_argument('--visualize', action='store_true')
    
    args, unknown = parser.parse_known_args()
    
    if args.config == None:
        print('Please input config file')
    else:
        print(f'Parsing config file: {args.config}')
    
    if args.generate:
        generator = CloudGenerator(args.config)
    
    if args.simulate:
        simulator = Simulator(args.config)
        
        simulator.run()
    
    if args.visualize:
        post_processor = PostProcessor(args.config)