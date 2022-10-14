from utils import unnorm, remove_axes
import matplotlib.pyplot as plt
from modules import *
import hydra
import cv2
import torch.multiprocessing
from PIL import Image
from utils import *
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
torch.multiprocessing.set_sharing_strategy('file_system')


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = join(root)
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(join(self.root, self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


@hydra.main(config_path="configs", config_name="demo_config.yml")
def my_app(cfg: DictConfig) -> None:
    device = torch.device("cuda:" + cfg.cuda_n if True else "cpu")
    print("device is set for: {}".format(device))
    result_dir = "/home/hossein/github/STEGO/results/predictions/truck/{}".format(
        cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(
        cfg.model_path, device=cfg.cuda_n)
    print(OmegaConf.to_yaml(model.cfg))

    dataset = UnlabeledImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.res, False, "center"),
    )

    loader = DataLoader(dataset, cfg.batch_size * 2,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().to(device)
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    for i, (img, name) in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = img.to(device)
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(
                code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(
                model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

            for j in range(img.shape[0]):
                single_img = img[j].cpu()
                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)
                new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                generate_image(model, single_img, linear_crf,
                               cluster_crf, join(result_dir, new_name))


def generate_image(model, img, linear_crf, cluster_crf, name):
    fig, ax = plt.subplots(1, 2, figsize=(5*5, 10))
    ax[0].imshow(unnorm(img).permute(1, 2, 0).cpu())
    ax[0].set_title("Image")
    # Removed cluster_crf for now
    # ax[2].imshow(model.label_cmap[cluster_crf])
    # ax[2].set_title("Cluster Predictions")
    ax[1].imshow(model.label_cmap[linear_crf])
    ax[1].set_title("Linear Probe Predictions")
    remove_axes(ax)
    fig.savefig(name)


if __name__ == "__main__":
    prep_args()
    my_app()
