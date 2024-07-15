# Introductio-Simple-LLM-Inference-on-CPU-and-finetuning-of-LLM-Model-to-create-a-Custom-Chatbot
# Gemma Fine-Tuning Project

This repository contains a Jupyter notebook for fine-tuning a language model using the Gemma dataset. The project aims to demonstrate the process of training, fine-tuning, and deploying a specialized chatbot.

## Project Overview

The goal of this project is to fine-tune a pre-trained language model to handle specific domain-related queries accurately. By using datasets relevant to the target domain, the chatbot can be tailored for various applications such as customer service or medical assistance.

## Features

### Data Cleaning and Preprocessing

Before training the model, the data needs to be cleaned and preprocessed. This involves:
- Removing any irrelevant or noisy data.
- Normalizing text to a consistent format.
- Tokenizing the text into a format suitable for the model.

### Fine-Tuning

Fine-tuning involves adjusting a pre-trained model on a specific dataset to improve its performance in a particular domain. This process includes:
- Loading a pre-trained model.
- Training the model on the domain-specific dataset.
- Validating the model's performance and making necessary adjustments.

### Optimization

Optimizing the model ensures that it runs efficiently on various hardware. Techniques include:
- Quantization: Reducing the precision of the model's weights to decrease memory usage and increase inference speed.
- Pruning: Removing less important parts of the model to reduce its size and improve performance.

### Interactive Interface

To enhance user interaction, the project uses frameworks like Streamlit or Chainlit to create an interactive chatbot interface. This allows users to:
- Interact with the chatbot in real-time.
- Provide feedback on the chatbot's responses.
- Visualize the chatbot's performance and make adjustments as needed.

### Continuous Improvement

Continuous improvement involves regularly updating the model based on user feedback and new data. This ensures that the chatbot remains accurate and relevant over time. Steps include:
- Collecting user feedback.
- Incorporating new data into the training set.
- Re-training and validating the model.

## Getting Started

### Prerequisites

To run the notebook and fine-tune the model, you need:
- Python 3.7 or higher
- PyTorch: A deep learning framework used for model training and inference.
- Huggingface Transformers: A library for working with pre-trained language models.
- Streamlit or Chainlit: Frameworks for creating interactive web applications.

### Installation

1. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Open the Jupyter notebook:

   ```bash
   jupyter notebook Gemma_Finetuning.ipynb
   ```

2. Follow the instructions in the notebook to preprocess the data, fine-tune the model, and deploy the chatbot interface.

### Example

Here's an example of generating responses with the fine-tuned model:

```python
text = "prompt: A number divided by 10 is 6. Yoongi got the result by subtracting 15 from a certain number. What is the result he got?"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Contributing

Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the creators of the pre-trained language models.
- Special thanks to the contributors of the datasets used for fine-tuning.
