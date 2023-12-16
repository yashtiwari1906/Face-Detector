import kserve
import argparse
from .detector_driver import DetectorModel

# curl --request POST 'http://localhost:8080/v2/models/dsnogoinspection/infer' --header "Content-Type: application/json" -d @./test_json_so.json

DEFAULT_MODEL_NAME = 'detector-model'
MODEL_DIR = '/mnt/models'
# MODEL_DIR = './'

parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
parser.add_argument('--model_dir', default=MODEL_DIR,
                    help='A URI pointer to the model directory')
parser.add_argument('--model_name', default=DEFAULT_MODEL_NAME,
                    help='The name that the model is served under.')

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    driver = DetectorModel(args.model_name, args.model_dir)
    kfserver = kserve.ModelServer()
    kfserver.start(models=[driver])
