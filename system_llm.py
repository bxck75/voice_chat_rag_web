import os
import time
import logging
from ctransformers import AutoModelForCausalLM

class SystemLLM:
    """A wrapper class for loading and using quantized LLMs for prompt generation"""
    
    def __init__(self, 
                 model_repo="TheBloke/Qwen1.5-1.8B-Chat-GGUF", 
                 model_file="qwen1_5-1_8b-chat.q4_k_m.gguf", 
                 gpu_layers=50,
                 cache_dir=None):
        """
        Initialize the LLM system
        
        Args:
            model_repo (str): HuggingFace repo containing the GGUF models
            model_file (str): Specific GGUF file to load
            model_type (str): Model architecture type (llama, falcon, etc.)
            gpu_layers (int): Number of layers to offload to GPU (0 for CPU-only)
            cache_dir (str): Directory to cache downloaded models
        """
        self.logger = logging.getLogger("SystemLLM")
        self.logger.info(f"Initializing LLM from {model_repo}/{model_file}")
        
        try:
            # Configure cache directory if provided
            kwargs = {}
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
                
            # Load the model
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_repo,
                model_file=model_file,
                model_type=model_type,
                gpu_layers=gpu_layers,
                **kwargs
            )
            
            self.logger.info("LLM loaded successfully")
            self.model_info = {
                "repo": model_repo,
                "file": model_file,
                "type": model_type,
                "gpu_layers": gpu_layers
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load LLM: {e}")
            raise RuntimeError(f"Failed to initialize LLM: {e}")
    
    def generate(self, prompt, max_tokens=512, temperature=0.7, top_p=0.95, top_k=40, repetition_penalty=1.1):
        """
        Generate text using the loaded LLM
        
        Args:
            prompt (str): Input prompt for the LLM
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature (higher = more random)
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            repetition_penalty (float): Penalty for repetition
            
        Returns:
            str: Generated text
        """
        try:
            start_time = time.time()
            
            # Generate text
            result = self.llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty
            )
            
            generation_time = time.time() - start_time
            self.logger.info(f"Generated {len(result.split())} words in {generation_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Error generating text: {e}"


class PromptGenerator:
    """Generates creative prompts for image generation using an LLM"""
    
    # Template for general image generation
    GENERAL_TEMPLATE = """
You are an expert prompt engineer for image generation. Create a detailed, creative, and visually descriptive prompt for an AI image generator based on the following concept: {concept}.

Your prompt should:
1. Be highly detailed and descriptive
2. Include specific visual elements, style, mood, lighting, and composition
3. Avoid any negative language or instructions (don't mention what not to include)
4. Be between 30-100 words

Prompt:
"""

    # Template for artistic image generation
    ARTISTIC_TEMPLATE = """
Create a detailed artistic prompt for an AI image generator based on this concept: {concept}

Include specific artistic styles, techniques, references to famous artists, composition details, color palettes, lighting effects, and mood.

Your prompt should be evocative, detailed, and precise. Focus on creating a cohesive artistic vision that could be realized as a striking visual image.

Prompt:
"""

    # Template for photorealistic image generation
    PHOTOREALISTIC_TEMPLATE = """
Create a photorealistic prompt for an AI image generator based on this concept: {concept}

Include camera specifications (lens focal length, aperture), lighting conditions, time of day, composition details, and subject positioning. Be extremely specific about lighting, materials, textures, and environmental factors.

Your prompt should read like a professional photography direction for a high-end photoshoot.

Prompt:
"""

    def __init__(self, llm_system=None, default_style="general"):
        """
        Initialize the prompt generator
        
        Args:
            llm_system (SystemLLM): An initialized LLM system
            default_style (str): Default prompt style template to use
        """
        self.logger = logging.getLogger("PromptGenerator")
        
        # Initialize LLM if not provided
        if llm_system is None:
            try:
                self.logger.info("Initializing default LLM system")
                self.llm = SystemLLM(
                    model_repo="TheBloke/Qwen1.5-1.8B-Chat-GGUF", 
                    model_file="qwen1_5-1_8b-chat.q4_k_m.gguf", 
                    model_type="llama", 
                    gpu_layers=50
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize default LLM: {e}")
                raise RuntimeError(f"Failed to initialize default LLM: {e}")
        else:
            self.llm = llm_system
            
        self.default_style = default_style
        self.templates = {
            "general": self.GENERAL_TEMPLATE,
            "artistic": self.ARTISTIC_TEMPLATE,
            "photorealistic": self.PHOTOREALISTIC_TEMPLATE
        }
    
    def generate(self, concept="", style=None, temperature=0.7):
        """
        Generate a prompt based on a concept
        
        Args:
            concept (str): The base concept or idea to build upon
            style (str): The style template to use (general, artistic, photorealistic)
            temperature (float): Creativity level (higher = more creative)
            
        Returns:
            str: Generated prompt
        """
        if not concept:
            concept = self._generate_random_concept()
            self.logger.info(f"Using generated concept: {concept}")
        
        style = style or self.default_style
        if style not in self.templates:
            self.logger.warning(f"Unknown style '{style}', falling back to default")
            style = self.default_style
            
        template = self.templates[style]
        full_prompt = template.format(concept=concept)
        
        self.logger.info(f"Generating {style}-style prompt for concept: {concept}")
        result = self.llm.generate(full_prompt, temperature=temperature)
        
        # Clean up the result - find the actual prompt part
        # For simple models, we might need to extract just the generated part
        clean_result = self._clean_result(result, full_prompt)
        
        return clean_result
    
    def _clean_result(self, result, input_prompt):
        """Extract just the generated prompt from the model output"""
        # Remove the input prompt part if it's included in the output
        if result.startswith(input_prompt):
            result = result[len(input_prompt):].strip()
            
        # Look for common markers that the model might add
        markers = ["Prompt:", "Generated Prompt:", "Image Prompt:"]
        for marker in markers:
            if marker in result:
                result = result.split(marker, 1)[1].strip()
                
        return result
    
    def _generate_random_concept(self):
        """Generate a random concept if none is provided"""
        concepts = [
            "a futuristic city at sunset",
            "an enchanted forest with magical creatures",
            "an underwater civilization",
            "a space station orbiting a distant planet",
            "a cyberpunk street market at night",
            "a medieval castle during a thunderstorm",
            "a peaceful mountain landscape",
            "a post-apocalyptic urban environment",
            "a steampunk laboratory with intricate machinery",
            "a desert oasis with exotic plants"
        ]
        
        import random
        return random.choice(concepts)



# Example usage
if __name__ == "__main__":
   
    # Initialize LLM system
    system_llm = SystemLLM(
        model_repo="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF", 
        model_file="qwen2.5-coder-3b-instruct-q4_k_m.gguf", 
        model_type="qwen", 
        gpu_layers=50,
        

    )
    
    # Create prompt generator
    prompt_gen = PromptGenerator(system_llm)
    
    # Generate sample prompts
    artistic_prompt = prompt_gen.generate("A mysterious library", style="artistic")
    print("Artistic Prompt:", artistic_prompt)
    