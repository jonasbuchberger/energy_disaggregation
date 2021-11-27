import os
from pathlib import Path

from tensorboard import program


def start_tensorboard(log_dir):
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f'TensorBoard started: {url}')
    input("Press Enter to end TensorBoard.")


ROOT_DIR = Path(__file__).parent.parent

if __name__ == '__main__':
    start_tensorboard(os.path.join(ROOT_DIR, 'logs'))
