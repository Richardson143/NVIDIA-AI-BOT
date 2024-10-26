
from openai import OpenAI
import streamlit as st
from datetime import datetime
import json
import time
import tiktoken
from crawl4ai import WebCrawler
import base64
import re
import os
import json
import requests
from PIL import Image
from io import BytesIO

class ChatbotConfig:
    def __init__(self):
        self.DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"
        self.MAX_TOKENS = 128000  # Maximum context window
        self.BATCH_SIZE = 4000 # Tokens per batch
        self.TEMPERATURE_RANGES = {
            'Conservative': 0.2,
            'Balanced': 0.4,
            'Creative': 0.6
        }
        self.PERSONA_PROMPTS = {
            'General Assistant': (
                "I am your friendly and versatile assistant, ready to provide clear and actionable support across a variety of topics. "
                "I can help you with: \n"
                "• Answering general questions in an informative and concise manner\n"
                "• Offering practical tips and resources for day-to-day tasks\n"
                "• Guiding you through decisions with thoughtful suggestions\n"
                "• Explaining complex ideas in a simple, easy-to-understand way\n"
                "Let me know how I can assist you today!"
            ),
            'Technical Expert': (
                "I am your expert technical companion, with deep expertise in software development, system architecture, and emerging technologies. "
                "I can help you with: \n"
                "• Writing and debugging code across multiple programming languages\n"
                "• Explaining complex technical concepts with practical examples\n"
                "• Providing system design recommendations and best practices\n"
                "• Troubleshooting technical issues with detailed step-by-step guidance\n"
                "• Staying updated with cutting-edge technology trends\n"
                "I emphasize clean code, scalable solutions, and industry best practices in all my responses. What technical challenge can I help you with?"
            ),
            'Academic Tutor': (
                "I am your patient and knowledgeable academic tutor, specializing in helping students grasp complex concepts, especially in STEM fields. "
                "I can assist you by: \n"
                "• Breaking down difficult subjects into simple, easy-to-follow explanations\n"
                "• Offering step-by-step walkthroughs for solving problems\n"
                "• Using real-world examples and analogies to clarify abstract ideas\n"
                "• Providing practice problems and solutions for deeper understanding\n"
                "How can I support your learning today?"
            ),
            'Creative Writer': (
                "I am a passionate creative writer skilled in crafting stories, poetry, and vivid descriptions. "
                "I can assist you with: \n"
                "• Writing captivating narratives with emotional depth\n"
                "• Creating rich metaphors, analogies, and vivid imagery\n"
                "• Developing unique characters, worlds, and plotlines\n"
                "• Helping with poetry, song lyrics, or other forms of artistic expression\n"
                "Let's collaborate on your next creative project!"
            ),
            'Business Consultant': (
                "I am an insightful business consultant with a focus on strategy, growth, and financial optimization. "
                "I can assist with: \n"
                "• Crafting effective business strategies for scaling and growth\n"
                "• Providing financial analysis and budgeting advice\n"
                "• Offering market insights and recommendations for business expansion\n"
                "• Assisting with operational improvements for efficiency and profitability\n"
                "How can I help you drive your business forward?"
            ),
            'Health & Wellness Coach': (
                "I am a holistic health and wellness coach, ready to guide you toward a balanced lifestyle. "
                "I can help you with: \n"
                "• Personalized workout routines and fitness plans\n"
                "• Nutrition advice tailored to your specific goals\n"
                "• Tips for maintaining mental well-being and reducing stress\n"
                "• Guidance on establishing healthy habits and routines\n"
                "What aspect of your health journey can I assist you with today?"
            ),
            'Legal Advisor': (
                "I am your trusted legal advisor, here to provide clear and practical legal guidance. "
                "I can help with: \n"
                "• Explaining legal concepts in an easy-to-understand way\n"
                "• Offering advice on contract law, intellectual property, and corporate law\n"
                "• Guiding you through legal decisions and ensuring compliance\n"
                "• Assisting with risk assessment and protection strategies\n"
                "Let me know how I can help with your legal questions!"
            ),
            'Project Manager': (
                "I am your organized and results-driven project manager, here to help you lead successful projects. "
                "I can assist with: \n"
                "• Developing project plans, timelines, and milestones\n"
                "• Offering guidance on agile methodologies and project management tools\n"
                "• Coordinating team efforts to ensure on-time delivery\n"
                "• Managing risks and communicating effectively with stakeholders\n"
                "What project can I help you plan and execute today?"
            ),
            'Language Translator': (
                "I am a skilled language translator, experienced in translating both technical and non-technical content. "
                "I can assist you with: \n"
                "• Translating text while preserving context, tone, and cultural nuances\n"
                "• Helping with multilingual communication, from emails to documents\n"
                "• Offering insights into linguistic subtleties between different languages\n"
                "What translation do you need help with today?"
            ),
            'Financial Advisor': (
                "I am a knowledgeable financial advisor, ready to assist with personal finance and investment strategies. "
                "I can help you with: \n"
                "• Creating and managing a budget tailored to your goals\n"
                "• Offering advice on saving, investing, and growing your wealth\n"
                "• Guiding you through retirement planning and debt management\n"
                "• Providing insights on smart investment opportunities\n"
                "How can I help you achieve financial success today?"
            ),
            'Motivational Coach': (
                "I am your personal motivational coach, here to inspire and empower you to reach your full potential. "
                "I can assist with: \n"
                "• Offering strategies to overcome obstacles and stay focused\n"
                "• Providing motivational tips to keep you energized and committed\n"
                "• Helping you build confidence and set achievable goals\n"
                "• Offering encouragement to help you stay positive and determined\n"
                "What goal are you working on today, and how can I support you?"
            ),
            'Travel Guide': (
                "I am your seasoned travel guide, with a wealth of knowledge on destinations, travel tips, and local experiences. "
                "I can assist you with: \n"
                "• Curating personalized travel itineraries based on your interests\n"
                "• Recommending hidden gems and must-visit spots around the world\n"
                "• Offering travel tips, from packing advice to navigating airports\n"
                "• Sharing local customs, traditions, and insider knowledge\n"
                "Where are you headed next, and how can I help you plan your trip?"
            ),
            'Life Coach': (
                "I am your thoughtful life coach, ready to help you navigate personal challenges and discover your true potential. "
                "I can help you with: \n"
                "• Setting meaningful goals and creating a plan to achieve them\n"
                "• Offering strategies for overcoming obstacles and self-doubt\n"
                "• Helping you cultivate self-awareness and personal growth\n"
                "• Providing insights on improving work-life balance and overall fulfillment\n"
                "How can I support your personal growth journey today?"
            ),
            'Parenting Expert': (
                "I am your compassionate parenting expert, with extensive knowledge in child development and family dynamics. "
                "I can assist with: \n"
                "• Offering practical advice for managing child behavior and discipline\n"
                "• Guiding you through developmental milestones for all age groups\n"
                "• Providing strategies for creating a positive and nurturing environment\n"
                "• Offering tips on parenting challenges, from bedtime routines to school issues\n"
                "What parenting challenge can I help you with today?"
            ),
            'Career Counselor': (
                "I am your experienced career counselor, here to help you navigate career transitions and opportunities. "
                "I can assist with: \n"
                "• Offering personalized advice on career planning and development\n"
                "• Guiding you through resume building, cover letters, and interview preparation\n"
                "• Providing insights on industry trends and skill development\n"
                "• Helping you find and pursue new career opportunities\n"
                "What career challenge or opportunity can I help you with today?"
            ),
            'Fitness Trainer': (
                "I am your dedicated fitness trainer, focused on helping you achieve your health and fitness goals. "
                "I can assist you with: \n"
                "• Creating customized workout plans based on your fitness level\n"
                "• Offering guidance on proper exercise form and technique\n"
                "• Providing nutritional advice to complement your fitness journey\n"
                "• Offering tips on staying motivated and consistent with your routine\n"
                "What are your fitness goals, and how can I support you today?"
            ),
            'Environmental Specialist': (
                "I am an expert in environmental science and sustainability, passionate about helping you make eco-friendly choices. "
                "I can assist with: \n"
                "• Offering advice on sustainable living practices and green technology\n"
                "• Helping you understand the environmental impact of human activities\n"
                "• Providing tips on waste reduction, energy efficiency, and conservation\n"
                "• Sharing insights on renewable energy and environmental protection\n"
                "How can I help you live more sustainably today?"
            ),
            'Entrepreneur Mentor': (
                "I am your experienced mentor, dedicated to helping aspiring entrepreneurs launch and grow successful businesses. "
                "I can help with: \n"
                "• Developing business ideas and crafting a viable business plan\n"
                "• Offering advice on funding, scaling, and managing a startup\n"
                "• Providing insights on market trends, competition, and growth strategies\n"
                "• Helping you navigate the challenges of entrepreneurship with practical solutions\n"
                "What part of your entrepreneurial journey can I assist you with today?"
            )
        }

def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

def download_markdown(content, filename="extracted_content.md"):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Download Markdown File</a>'
    return href

# Constants for image processing
INVOKE_URL = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct/chat/completions"
STREAM = True

def compress_image(image_file, max_size_kb=175):
    """Compress the uploaded image to meet size requirements"""
    max_size_bytes = max_size_kb * 1024
    quality = 95
    
    img = Image.open(image_file)
    img.thumbnail((800, 800))
    
    while True:
        img_byte_arr = BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=quality)
        if img_byte_arr.tell() <= max_size_bytes or quality <= 10:
            return img_byte_arr.getvalue()
        quality = max(quality - 10, 10)

def process_image(image_file, api_key, question):
    """Process the image and get response from the vision model"""
    try:
        compressed_image = compress_image(image_file)
        image_b64 = base64.b64encode(compressed_image).decode()
        
        if len(image_b64) >= 180_000:
            return "Error: Image is still too large after compression. Please try a smaller image."
            
        if not api_key:
            api_key = os.getenv("YOUR_API_KEY")
            
        prompt = f"{question}"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream" if STREAM else "application/json"
        }
        
        payload = {
            "model": 'meta/llama-3.2-90b-vision-instruct',
            "messages": [
                {
                    "role": "user",
                    "content": f'{prompt} <img src="data:image/jpeg;base64,{image_b64}" />'
                }
            ],
            "max_tokens": 512,
            "temperature": 1.00,
            "top_p": 1.00,
            "stream": STREAM
        }

        with st.spinner('Analyzing image...'):
            response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=True)
            
            if response.status_code == 200:
                full_response = ""
                response_placeholder = st.empty()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            json_str = line[6:]
                            if json_str.strip() == '[DONE]':
                                break
                            try:
                                json_obj = json.loads(json_str)
                                content = json_obj['choices'][0]['delta'].get('content', '')
                                full_response += content
                                response_placeholder.write(full_response)
                            except json.JSONDecodeError:
                                st.error(f"Failed to parse JSON: {json_str}")
                
                return full_response
                
            elif response.status_code == 402:
                return "Error: API account credits have expired. Please check your account status on the NVIDIA website."
            else:
                error_message = f"Error {response.status_code}: {response.text}"
                st.error(error_message)
                return f"An error occurred. Please try again later or contact support. Error code: {response.status_code}"
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"Error processing request: {str(e)}"
    
class ResponseManager:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.config = ChatbotConfig()
    
    def count_tokens(self, text: str) -> int:
        """Approximate token count using the appropriate tokenizer for the NVIDIA model."""
        try:
            # Assuming you have a function or library that provides the correct tokenization for the NVIDIA model
            encoding = tiktoken.encoding_for_model("nvidia/llama-3.1-nemotron-70b-instruct")  # Use the correct model
            return len(encoding.encode(text))
        except Exception:
            # Fallback to word-based approximation
            return len(text.split()) * 1.3  # Adjust this as necessary for a better approximation
    
    def generate_response(self, messages, temperature, placeholder):
        """Generate response with continuation handling in batches."""
        full_response = ""
        continuation_prompt = "\nPlease continue from where you left off..."
        current_messages = messages.copy()
        
        try:
            while True:
                # Calculate remaining tokens
                remaining_tokens = self.config.MAX_TOKENS - self.count_tokens(full_response)
                tokens_to_generate = min(self.config.BATCH_SIZE, remaining_tokens)

                # Generate response in batches
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=current_messages,
                    temperature=temperature,
                    max_tokens=tokens_to_generate,
                    stream=True
                )
                
                batch_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        chunk_content = chunk.choices[0].delta.content
                        batch_response += chunk_content
                        full_response += chunk_content
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                
                # Check if response seems complete
                if batch_response.strip().endswith((".", "!", "?", "\n")) or \
                len(batch_response.strip()) < tokens_to_generate * 0.9:
                    break
                
                # Prepare for continuation
                current_messages.append({"role": "assistant", "content": full_response})
                current_messages.append({"role": "user", "content": continuation_prompt})
            
            return full_response
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return f"Error generating response: {str(e)}"

def initialize_session_state():
    """Initialize all session state variables"""
    if "messages" not in st.session_state:
        # Set the initial system message based on the default persona
        initial_persona = ChatbotConfig().PERSONA_PROMPTS['General Assistant']
        st.session_state.messages = [{"role": "system", "content": initial_persona}]
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "nvidia_model" not in st.session_state:
        st.session_state.nvidia_model = ChatbotConfig().DEFAULT_MODEL
    if "image_mode" not in st.session_state:
        st.session_state.image_mode = False

def load_conversations():
    """Load conversation history from JSON files."""
    conversation_files = [f for f in os.listdir() if f.startswith('chat_history_') and f.endswith('.json')]
    return conversation_files

def load_conversation(file_name):
    """Load a specific conversation from a JSON file."""
    with open(file_name, 'r') as f:
        return json.load(f)

def save_conversation(filename="chat_history.json"):
    """Save the current conversation to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_history_{timestamp}.json"
    
    conversation_data = {
        "timestamp": timestamp,
        "messages": st.session_state.messages[1:],  # Exclude system message
    }
    
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=2)

    return filename

def create_sidebar():
    """Create and handle sidebar elements"""
    config = ChatbotConfig()
    
    st.sidebar.title("NVIDIA NIM Chatbot ⚙️")

    # Web Scraping Toggle
    st.sidebar.header("Web Scraping")
    enable_web_scraping = st.sidebar.toggle("Enable Automatic Web Scraping", value=False)
    
    if enable_web_scraping:
        st.sidebar.info("URLs detected in your input will be automatically scraped for additional context by using Crawl4AI.")

    # Model Settings
    st.sidebar.header("Model Configuration")
    
    # Persona Selection
    personas = list(config.PERSONA_PROMPTS.keys())
    
    # Add custom persona input fields within an expander
    with st.sidebar.expander("✨ Create Custom Persona"):
        st.markdown("### Create Your Own Persona")
        custom_persona_name = st.text_input("Persona Name", 
            placeholder="e.g., Data Science Expert, Marketing Specialist")
        custom_persona_description = st.text_area("Persona Description", 
            placeholder="Describe the persona's expertise, tone, and capabilities...",
            height=150)
        
        if st.button("Add Custom Persona", type="primary"):
            if custom_persona_name and custom_persona_description:
                # Add the custom persona to the list of personas
                config.PERSONA_PROMPTS[custom_persona_name] = custom_persona_description
                st.success(f"✅ Custom persona '{custom_persona_name}' added successfully!")
                time.sleep(1)  # Show success message briefly
                st.rerun()  # Refresh to update the persona list
            else:
                st.error("Please provide both a name and a description for the custom persona.")
    
    # Update persona selection with custom personas
    selected_persona = st.sidebar.selectbox(
        "Choose Assistant Persona",
        list(config.PERSONA_PROMPTS.keys()),
        help="Select from pre-defined personas or create your own custom persona"
    )
    
    # Display current persona description
    with st.sidebar.expander("Current Persona Description", expanded=False):
        st.markdown(f"### {selected_persona}")
        st.markdown(config.PERSONA_PROMPTS[selected_persona])
    
    # Response Style
    temperature_style = st.sidebar.selectbox(
        "Response Style",
        list(config.TEMPERATURE_RANGES.keys())
    )
    
    # Advanced Settings Expander
    with st.sidebar.expander("⚙️ Advanced Settings"):
        show_token_count = st.checkbox("Show Token Count", value=False)
        enable_code_highlighting = st.checkbox("Enable Code Highlighting", value=True)
        enable_markdown = st.checkbox("Enable Markdown Support", value=True)
        batch_size = st.slider("Response Batch Size (tokens)", 
                             min_value=100, 
                             max_value=4000, 
                             value=1000, 
                             step=100)
    
    # Image Chat Mode Toggle
    st.sidebar.header("🤖 Llama 3.2 90B Vision Analysis")
    image_mode = st.sidebar.toggle("Enable Image Chat", value=st.session_state.image_mode)
    st.session_state.image_mode = image_mode
    
    if image_mode:
        st.sidebar.info("Image chat mode is enabled. You can now upload images and ask questions about them.")

    # Load previous conversations
    st.sidebar.header("Load Previous Conversations")
    conversation_files = load_conversations()
    if conversation_files:
        selected_file = st.sidebar.selectbox("Choose a conversation to load", conversation_files)
        if st.sidebar.button("Load Conversation"):
            conversation_data = load_conversation(selected_file)
            st.session_state.messages = conversation_data['messages']
            st.success("Conversation loaded successfully!")
            st.experimental_rerun()  # Refresh to display loaded conversation
    else:
        st.sidebar.info("No previous conversations found.")

    # Conversation Management
    st.sidebar.header("Conversation Management")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [{"role": "system", "content": config.PERSONA_PROMPTS[selected_persona]}]
            st.rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            conversation_json = save_conversation()
            st.download_button(
                label="📥 Download",
                data=conversation_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    
    return {
        'temperature': config.TEMPERATURE_RANGES[temperature_style],
        'batch_size': batch_size,
        'show_token_count': show_token_count,
        'enable_code_highlighting': enable_code_highlighting,
        'enable_markdown': enable_markdown,
        'persona': config.PERSONA_PROMPTS[selected_persona],
        'enable_web_scraping': enable_web_scraping
    }

def format_message(message, enable_code_highlighting=True, enable_markdown=True):
    """Format message with optional code highlighting and markdown support"""
    if message["role"] == "system" and message["content"].startswith("Additional context from web scraping:"):
        # Don't display system messages with scraped content in the chat window
        return
    
    content = message["content"]
    
    if enable_code_highlighting and "```" in content:
        # Enhanced code block detection and formatting
        parts = content.split("```")
        formatted_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Code block
                try:
                    lang, code = part.split("\n", 1)
                    formatted_parts.append(f'<div class="code-block {lang}">\n{code}\n</div>')
                except ValueError:
                    formatted_parts.append(f'<div class="code-block">\n{part}\n</div>')
            else:  # Regular text
                formatted_parts.append(part)
        content = "".join(formatted_parts)
    
    if enable_markdown:
        st.markdown(content, unsafe_allow_html=True)
    else:
        st.write(content)

def main():
    st.title("🧠 BrainWave AI IntelliChat 🤖")
    st.markdown("<h3 style='text-align: center;'>Powered by Llama 3.1 Nemotron-70B</h3>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    initialize_session_state()
    
    # Setup sidebar and get settings
    settings = create_sidebar()
    
    # Initialize OpenAI client and ResponseManager
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )
    response_manager = ResponseManager(client, st.session_state.nvidia_model)
    
    # Display chat history
    for message in st.session_state.messages[1:]:  # Skip system message
        if message["role"] != "system" or not message["content"].startswith("Additional context from web scraping:"):
            with st.chat_message(message["role"]):
                format_message(
                    message,
                    enable_code_highlighting=settings['enable_code_highlighting'],
                    enable_markdown=settings['enable_markdown']
                )
    
    # Initialize session state for expander visibility
    if 'show_scraped_content' not in st.session_state:
        st.session_state.show_scraped_content = False

    if st.session_state.image_mode:
        # Image chat mode
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
            if uploaded_file:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            api_key = st.text_input("Enter your API Key", type="password", 
                                placeholder="API authentication key")
            question = st.text_input("Enter your question", 
                                placeholder="Example: What is in this image?")
            
            if st.button("Analyze Image", use_container_width=True):
                if uploaded_file and question:
                    response = process_image(uploaded_file, api_key, question)
                    st.markdown("### Analysis Result:")
                    st.markdown(response)
                else:
                    st.warning("Please upload an image and enter a question.")
    else:    
        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            if settings['enable_web_scraping']:
                urls = extract_urls(prompt)
                if urls:
                    scraped_contents = {}
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for i, url in enumerate(urls):
                        status_text.text(f"Scraping URL {i+1}/{len(urls)}: {url}")
                        try:
                            crawler = WebCrawler()
                            crawler.warmup()
                            result = crawler.run(url=url)
                            scraped_contents[url] = result.markdown
                            st.sidebar.success(f"Scraped content from: {url}")
                            st.sidebar.markdown(download_markdown(result.markdown, f"content_from_{url.replace('://', '_')}.md"), unsafe_allow_html=True)
                        except Exception as e:
                            st.sidebar.error(f"Error scraping {url}: {str(e)}")
                        progress_bar.progress((i + 1) / len(urls))

                    status_text.text("Scraping completed!")
                    progress_bar.empty()

                    if scraped_contents:
                        # Create checkboxes for each URL
                        st.write("Select URLs to include in the context:")
                        url_selections = {url: st.checkbox(f"Include {url}", value=True) for url in scraped_contents.keys()}

                        # Combine selected scraped contents
                        selected_contents = "\n\n".join([f"Content from {url}:\n{content}" 
                                                        for url, content in scraped_contents.items() 
                                                        if url_selections[url]])

                        # Add selected scraped content as a system message (hidden from chat window)
                        st.session_state.messages.append({"role": "system", "content": f"Additional context from web scraping:{selected_contents}"})
                        
                        # Store scraped contents in session state
                        st.session_state.scraped_contents = scraped_contents

                        # Set flag to show scraped content
                        st.session_state.show_scraped_content = True

            # Check if the prompt is a question about identity or help
            if "who are you" in prompt.lower() or "how can you help" in prompt.lower():
                # Respond based on the selected persona prompt
                persona_response = settings['persona']
                st.session_state.messages.append({"role": "assistant", "content": persona_response})
                with st.chat_message("assistant"):
                    st.markdown(persona_response)
            else:
                # Append the selected persona prompt to the messages
                selected_persona_prompt = settings['persona']
                st.session_state.messages.append({"role": "system", "content": selected_persona_prompt})

                # Generate and display assistant response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    
                    # Generate response with continuation handling
                    full_response = response_manager.generate_response(
                        messages=st.session_state.messages,
                        temperature=settings['temperature'],
                        placeholder=message_placeholder
                    )
                    
                    # Final update
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant response to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                    # Display token count if enabled
                    if settings['show_token_count']:
                        token_count = response_manager.count_tokens(full_response)
                        st.caption(f"Approximate tokens: {token_count}")
        
        # Display scraped content expander (outside the if prompt block)
        if st.session_state.get('show_scraped_content', False):
            with st.expander("View Scraped Content", expanded=False):
                if 'scraped_contents' in st.session_state and st.session_state.scraped_contents:
                    selected_url = st.selectbox("Choose URL to view content:", list(st.session_state.scraped_contents.keys()))
                    st.markdown(st.session_state.scraped_contents[selected_url])
                else:
                    st.write("No scraped content available.")

if __name__ == "__main__":
    main()

st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    /* Header styling */
    .title-container {
        background: linear-gradient(45deg, #2193b0, #6dd5ed);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .main-title {
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }

    /* Chat container styling */
    .chat-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }

    /* Message styling */
    .stTextInput>div>div>input {
        border-radius: 25px !important;
        border: 2px solid #e0e0e0;
        padding: 1rem 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stTextInput>div>div>input:focus {
        border-color: #2193b0;
        box-shadow: 0 0 0 2px rgba(33, 147, 176, 0.2);
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(45deg, #2193b0, #6dd5ed);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 2rem 1rem;
    }

    /* Feature cards */
    .feature-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }

    /* Creator section styling */
    .creator-section {
        background: linear-gradient(45deg, #141e30, #243b55);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        text-align: center;
    }

    .social-links a {
        color: #6dd5ed;
        text-decoration: none;
        margin: 0 1rem;
        transition: all 0.3s ease;
    }

    .social-links a:hover {
        color: white;
        text-decoration: none;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
""", unsafe_allow_html=True)

# Modified welcome section
# st.markdown("""
#     <div class="title-container animate-fade-in">
#         <h1 class="main-title">✨ Welcome to NVIDIA AI Chat Magic! ✨</h1>
#         <p style="color: white; font-size: 1.2rem;">Experience the future of AI conversation</p>
#     </div>

#     <div class="feature-card animate-fade-in">
#         <h2 style="color: #2193b0; margin-bottom: 1rem;">🎭 Discover Our Amazing Features</h2>
#         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
#             <div class="feature-item">
#                 <h3>🤖 AI Companions</h3>
#                 <p>Engage with personalized AI assistants</p>
#             </div>
#             <div class="feature-item">
#                 <h3>🌐 Web Integration</h3>
#                 <p>Access real-time web content</p>
#             </div>
#             <div class="feature-item">
#                 <h3>🖼️ Image Analysis</h3>
#                 <p>Intelligent image processing</p>
#             </div>
#             <div class="feature-item">
#                 <h3>🎨 Creative Control</h3>
#                 <p>Customize response styles</p>
#             </div>
#         </div>
#     </div>
# """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.title("✨ About the Creator")
st.sidebar.markdown("""
    <div style="font-family: 'Brush Script MT', cursive; font-size: 20px; color: #4A90E2;">
        Crafted with ❤️ by Richardson Gunde
    </div>
    
    <div style="font-family: 'Dancing Script', cursive; font-size: 16px; padding: 10px 0;">
        Featuring:
        <br>• ✨ Custom AI Personas
        <br>• 🌐 Web Content Integration
        <br>• 🖼️ Image Analysis
        <br>• 🎨 Creative Response Control
        <br>• 📊 Token Tracking
        <br>• 💭 Smart Conversations
    </div>
    
    <div style="font-family: 'Dancing Script', cursive; font-size: 16px; padding-top: 10px;">
        🔗 <a href="https://www.linkedin.com/in/richardson-gunde" style="color: #0077B5;">LinkedIn</a>
        <br>📧 <a href="mailto:gunderichardson@gmail.com" style="color: #D44638;">Email</a>
    </div>
    """, unsafe_allow_html=True)
