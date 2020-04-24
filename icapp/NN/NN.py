import os
import torch
import PIL
from PIL import Image
from torchvision import transforms
from vocabulary import Vocabulary


from model import (EncoderCNN,
                   DecoderRNN,
                  )


path_encoder_weight = os.path.join(os.getcwd(), "icapp", "NN", "models/encoder-4.pkl")
path_decoder_weight = os.path.join(os.getcwd(), "icapp", "NN", "models/decoder-4.pkl")
path_vocab_file = os.path.join(os.getcwd(), "icapp", "NN", "vocab.pkl")

# load Voacb
vocab = Vocabulary(vocab_threshold=5,
                   vocab_file=path_vocab_file,
                   vocab_from_file=True,
                  )

# define image transforms
transform_inference = transforms.Compose(
    [transforms.Resize(256),                          # smaller edge of image resized to 256
     transforms.CenterCrop(224),                      # get 224x224 crop from center location
     transforms.ToTensor(),                           # convert the PIL Image to a tensor
     transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                          (0.229, 0.224, 0.225))])

# define NN params
hidden_size = 512
embed_size = 256

# get device
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device("cpu")

# init 2 NN
encoder = EncoderCNN(embed_size)
encoder.eval()

decoder = DecoderRNN(embed_size, hidden_size, vocab_size=len(vocab))
decoder.eval()

# load weights
encoder.load_state_dict(torch.load(path_encoder_weight, map_location=DEVICE))
decoder.load_state_dict(torch.load(path_decoder_weight, map_location=DEVICE))

def get_image_captioning(image):

    if not isinstance(image, PIL.Image.Image):
        image = Image.open(image)
        image = image.convert("RGB")

    img = transform_inference(image)
    img = img.unsqueeze(0).to(DEVICE)

    # get feature map
    img = encoder(img)
    # get describe of image
    output = decoder.sample(img.unsqueeze(1))

    # Convert numbers in words
    caption = " ".join([vocab.idx2word[idx] for idx in output])

    return caption

def main():
    
    print("Wooooops! NN.py was launched")


if __name__ == "__main__":
    main()
