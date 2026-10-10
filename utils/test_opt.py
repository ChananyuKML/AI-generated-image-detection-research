import argparse
import os

class TestOptions():
    def __init__(self):
        self.initialized = False
    
    def initialize(self,parser):
        parser.add_argument('--generator', default="sd4")
        parser.add_argument('--network', default="dual_cnn")
        parser.add_argument('--loader', default="gen_image")
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_workers', type=int, default=12,
                            help='DataLoader workers; use 0 for a local smoke test')
        parser.add_argument('--checkpoint', default="100")
        parser.add_argument('--dataroot', type=str, default="dataset/genimage")
        parser.add_argument('--transform', default="hog_224")
        self.initialized = True
        return parser
    
    def collect_opts(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        self.parser = parser
        return parser.parse_args()
    
    def parse(self):
        opt = self.collect_opts()
        self.opt = opt
        return self.opt
