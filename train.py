from kbgen.Trainer import Trainer
from kbgen.config import defaults_customLM as config
from kbgen.utils.cli import parse_args


if __name__ == "__main__":
    # parse_args parses the command line arguments and returns a dictionary
    # the defaults for the arguments are given as a dictionary below
    # they are overwritten by the command line arguments of the same name
    config = parse_args(config)

    trainer = Trainer(config)
    trainer.train()
