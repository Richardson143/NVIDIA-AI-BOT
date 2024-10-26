# 🧠 BrainWave AI IntelliChat 

![image](https://github.com/user-attachments/assets/f038f32c-17de-4cfc-bc3d-abf0cf5a61ad)


An advanced AI chatbot powered by NVIDIA's Llama 3.1 Nemotron-70B and Llama 3.2 90B Vision models, featuring intelligent web scraping, image analysis, and dynamic persona management.

## 🌟 Features

### Core Capabilities
- 🤖 **Dual Model Integration**
  - NVIDIA Llama 3.1 Nemotron-70B for text processing
  - NVIDIA Llama 3.2 90B Vision for image analysis
- 🎭 **Dynamic Persona System**
  - 15+ pre-configured AI personalities
  - Custom persona creator
  - Persona-specific conversation styles
- 🌐 **Intelligent Web Scraping**
  - Real-time URL content extraction using Crawl4AI
  - Automatic context integration
  - Downloadable scraped content in Markdown
- 🖼️ **Image Analysis Mode**
  - Upload and analyze images
  - Natural language image questioning
  - Vision model integration

### Technical Features
- 🔄 **Smart Token Management**
  - Configurable batch sizes
  - Continuous response generation
  - Context preservation for long conversations
- 💾 **Conversation Management**
  - Save/load conversation histories
  - JSON export functionality
  - Conversation clearing options
- ⚙️ **Advanced Configuration**
  - Adjustable response temperatures
  - Token count display
  - Code highlighting options
  - Markdown support

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.8
streamlit >= 1.28
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/brainwave-intellichat.git
cd brainwave-intellichat
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
# Create a .env file in the project root
NVIDIA_API_KEY=your_nvidia_api_key
```

4. Run the application:

streamlit run app.py

## 🔧 Configuration

### Environment Variables
.env
NVIDIA_API_KEY=your_api_key
MAX_TOKENS=128000
BATCH_SIZE=4000


### Persona Configuration
Create custom personas by adding to `config/chatbot_config.py`:
python
PERSONA_PROMPTS = {
    "Custom_Persona": """
    Your custom persona description here
    """
}


## 🎯 Usage Examples

### Basic Chat
python
# Initialize chat with default settings
response = chatbot.generate_response(
    "Tell me about artificial intelligence",
    temperature=0.4
)


### Image Analysis
python
# Upload and analyze an image
response = chatbot.analyze_image(
    image_path="path/to/image.jpg",
    question="What can you see in this image?"
)


### Web Scraping
python
# Enable web scraping for URLs in conversation
chatbot.enable_web_scraping = True
response = chatbot.generate_response(
    "Analyze this article: https://example.com/article"
)


## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## 🙏 Acknowledgments

- NVIDIA for their amazing LLM models
- Crawl4AI for web scraping capabilities
- Streamlit for the wonderful web framework
- The open-source community for various tools and libraries
)

