import cv2
import logging
import torch
import hydra
from omegaconf import OmegaConf
from srcs.utils import instantiate
from datetime import datetime
import torchvision.transforms as transforms
import os

def create_folder_dictionary(folder_path):
    folder_dict = {}
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    for index, folder_name in enumerate(folders):
        folder_dict[index] = folder_name
    return folder_dict


logger = logging.getLogger('evaluate')

@hydra.main(version_base=None,config_path='conf', config_name='evaluate')
def main(config):
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint, map_location=torch.device('cpu'))

    loaded_config = OmegaConf.create(checkpoint['config'])

    # restore network architecture
    model = instantiate(loaded_config.arch)
    logger.info(model)

    # load trained weights
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    capture_and_process(model=model)

    return None



def capture_and_process(model):
    def get_log_filename():
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"log_{current_time}.txt"

    def process_output(output):
        return output.replace("space", " ").replace("del", "")

    img_id = 0
    shot = 1
    result = []
    capture = cv2.VideoCapture(0)

    dictionary = create_folder_dictionary("data/our_sign_language_dataset/")
    
    # Создаем папку captures с текущей датой
    current_date = datetime.now().strftime("%Y-%m-%d")
    captures_folder = os.path.join("captures", current_date)
    os.makedirs(captures_folder, exist_ok=True)

    log_filename = os.path.join(captures_folder, get_log_filename())
    with open(log_filename, "w") as log_file:
        while True:
            image = capture.read()[1]
            cv2.imshow('SMILE FACE', image)
            k = cv2.waitKey(30) & 0xFF
            if k == 27:  # escape
                break
            if not shot % 30:
                transform = transforms.ToTensor()
                filename = os.path.join(captures_folder, f"capture_{img_id}.jpg")
                cv2.imwrite(filename, image)
                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0)
                output = model(torch.Tensor(image_tensor).float()).argmax(dim=1)
                output = dictionary[output.item()] 
                if output == "del" and img_id > 0:
                    result.pop(img_id-1)
                result.append(process_output(output))
                img_id += 1
                log_file.write(f"Processed image {filename} at {datetime.now()}\n")
            shot += 1

    capture.release()
    cv2.destroyAllWindows()

    output_file = os.path.join(captures_folder, "output.txt")
    with open(output_file, "w") as file:
        file.write("".join(list(dict.fromkeys(result))))

# Call the function to execute
        
if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
