# QR Code Decoder Using Deep Learning

This project involves the creation of a QR code decoder built entirely on deep learning—a complex task that pushes the boundaries of conventional image processing techniques. Unlike typical image classification tasks, decoding QR codes presents unique challenges due to the intricate encoding mechanisms involved.

## Why is it challenging?

QR codes contain several elements that introduce noise, making it difficult for AI models to decode them accurately:

1. **Error Correction Codewords**: Reed-Solomon error correction adds redundant data to the QR code, creating noise that has no direct mathematical relationship with the encoded data.
2. **Data Masking**: This process alters bits of the original binary data to improve readability, further complicating the decoding process.
3. **Timing Patterns**: These patterns help scanners determine the QR code's size but add another layer of complexity for the AI to decipher.

## How Deep Learning Enhances QR Code Decoding

The deep learning model developed in this project, based on the EfficientNet family and tuned using AutoKeras, brings significant improvements over traditional QR code decoders, especially when dealing with damaged or partially obscured codes. 

### Key Advantages:

1. **Robustness to Damage**: Traditional QR code decoders rely heavily on the structured patterns of a QR code and can fail when those patterns are disrupted by serious damages or occlusion. The deep learning model, however, learns to recognize the underlying data patterns even in the presence of noise and distortion, allowing it to decode QR codes that would be unreadable by conventional methods.

2. **Noise Handling**: The model has been trained to account for the noise introduced by error correction, data masking, and timing patterns, enabling it to focus on the meaningful parts of the QR code data and disregard the irrelevant noise.

## Performance and Future Work

Despite the challenges, the best model achieved a cross-validated mean absolute error of 21,000—a starting point that highlights the potential for further improvements. The model, which has a size of 90MB, required approximately 7 hours of GPU computation time to train. 

Future work will focus on optimizing the model's accuracy and efficiency, as well as exploring additional deep learning architectures and training techniques. 

I'm also excited to announce that I’ll be releasing a YouTube video visualizing the architecture of this deep learning model in 3D, offering a deeper insight into its inner workings.

The model can be found here;

[Kaggle](https://www.kaggle.com/models/adedapoadeniran/ai-qr-code-decoder-deeplearning)

Stay tuned for more updates!
