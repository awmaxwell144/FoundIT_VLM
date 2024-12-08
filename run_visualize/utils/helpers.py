import os 
from PIL import Image
import logging
def copy_task_fitness(env_name):
    ROOT_DIR = os.getcwd()
    env_tff = f'{ROOT_DIR}/envs/{env_name}/task_fitness.py'
    output_tff = f'{ROOT_DIR}/run_visualize/scripts/tff.py'
    # Open the input file in read mode
    with open(env_tff, 'r') as input_file:
        tff = input_file.read()  # Read the content
    
    # Open the output file in write mode and write the content
    with open(output_tff, 'w') as output_file:
        output_file.write(tff)

def load_config(config_fname, seed_id=0, lrate=None):
    """Load training configuration and random seed of experiment."""
    import yaml
    import re
    from dotmap import DotMap

    def load_yaml(config_fname: str) -> dict:
        """Load in YAML config file."""
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        with open(config_fname) as file:
            yaml_config = yaml.load(file, Loader=loader)
        return yaml_config

    config = load_yaml(config_fname)
    config["train_config"]["seed_id"] = seed_id
    if lrate is not None:
        if "lr_begin" in config["train_config"].keys():
            config["train_config"]["lr_begin"] = lrate
            config["train_config"]["lr_end"] = lrate
        else:
            try:
                config["train_config"]["opt_params"]["lrate_init"] = lrate
            except Exception:
                pass
    return DotMap(config)


def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    import pickle
    from pathlib import Path

    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")


def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj


def extract_frames(env_name):
    """
    Extracts frames from a GIF and saves them as images in the specified folder.
    
    Args:
        gif_path (str): Path to the input GIF file.
        output_folder (str): Path to the folder where extracted frames will be saved.
    """
   
    ROOT_DIR = os.getcwd()

    output_folder = ROOT_DIR + "/evaluate/frames/"
    gif_path = ROOT_DIR + f"/output/{env_name}.gif"
     # Open the GIF file
    gif = Image.open(gif_path)
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    frame_number = 0
    try:
        while True:
            # Save the current frame as an image
            frame_path = os.path.join(output_folder, f"frame_{frame_number:03d}.png")
            gif.save(frame_path, "PNG")
            logging.debug(f"Saved frame {frame_number} to {frame_path}")
            frame_number += 1
            
            # Move to the next frame
            gif.seek(frame_number)
    except EOFError:
        logging.debug("End of gif")
        # End of GIF



    if __name__ == "__main__":
        extract_frames('CartPole-v1')