import sys
from datetime import datetime

from tagger.utils.logging import get_logger


class Tagger:
    """
    Main class running Tagger application
    """

    def __init__(self, arguments):
        self.logger = get_logger(__name__)
        self.mode = arguments.mode

    def main(self):
        """
        Main processing
        """
        self.logger.info('Tagger')
        self.logger.info('Mode =', self.mode)
        start_time = datetime.now()

        # Train
        if self.mode == 'train':
            # Read wav, compute melspectrogram and save it to .npy

            # Compile model

            # Read .npy and fit model

            # Compare new model to current one, replace if better

            pass

        # Predict
        elif self.mode == 'predict':
            # Read wav, compute melspectrogram

            # get model

            # Inference

            pass



        # End
        now = datetime.now()
        time_run = (now - start_time).total_seconds()
        self.logger.info('Processing time: {} seconds'.format(time_run))


def main():
    """main"""
    args = parse_args(args=sys.argv[1:])
    Topper(args).main()


if __name__ == '__main__':
    main()
