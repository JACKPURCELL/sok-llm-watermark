from . import generate_embeddings
from . import train_watermark_model
from . import generate_mappings
from . import watermark_and_detect
from . import watermark
from .watermark import WatermarkLogitsProcessor, WatermarkWindow, WatermarkContext
import torch

def prepare_watermark_model(embedding_input_path, embedding_output_path, model_path, size, output_model, watermark_model_epochs, watermark_model_lr, input_dim, mapping_length, mapping_output_dir):
    print("Generate embeddings for training watermark model")
    args = {"input_path": embedding_input_path,
            "output_path": embedding_output_path,
            "model_path": model_path,
            "size": size,}
    generate_embeddings.main(args)

    print("Train watermark model")
    args = {"input_path": embedding_output_path,
            "output_model": output_model,
            "input_dim": input_dim,
            "epochs": watermark_model_epochs,
            "lr": watermark_model_lr,}
    train_watermark_model.main(args)

    print("Generate mapping files")
    args = {"length": mapping_length,
            "output_dir": mapping_output_dir,}
    generate_mappings.main(args)

class aiwei23b_WatermarkDetector():
    def __init__(self, watermark_type, window_size, tokenizer, chunk_size, delta, transform_model, embedding_model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if watermark_type == "window": # use a window of previous tokens to hash, e.g. KGW
            self.watermark_model = WatermarkWindow(device, window_size, tokenizer)
            self.logits_processor = WatermarkLogitsProcessor(self.watermark_model)
        elif watermark_type == "context":
            self.watermark_model = WatermarkContext(device, chunk_size, tokenizer, delta = delta, transform_model_path=transform_model, embedding_model=embedding_model)
            self.logits_processor = WatermarkLogitsProcessor(self.watermark_model)
    
    def detect(self, text: str = None, **kwargs):
        z_score = self.watermark_model.detect(kwargs["prompt"]+text)
        output_dict = {}
        output_dict["z_score"] = z_score
        return output_dict
    
    def dummy_detect(self, text: str = None, **kwargs):
        output_dict = {}
        output_dict["z_score"] = 0
        return output_dict
    
    
    

