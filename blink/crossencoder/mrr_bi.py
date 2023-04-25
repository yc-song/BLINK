import os
import torch
import sys
sys.path.append('.')

from blink.common.params import BlinkParser

def main(params):
   fname = os.path.join(params["data_path"], "train.t7")
   train_data = torch.load(fname)
   label_input = train_data["labels"]
   print(label_input.shape)
if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__

    main(params)